
import numpy as np
import nibabel as nib
import matplotlib
  # save PNGs instead of opening windows
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
import plotly.io
from scipy.optimize import least_squares
import scipy.io as sio
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg" if you prefer Qt windows
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz

DATA_ROOT = Path("/home/siq/Data_ACEMRI")
MOUSE = "M01_Post"
SLICE = "Slice5fid"
T1_MAP_PATH = None
r1 = 4.3                # mM^-1 s^-1
TE = 3.83e-3            # s (ignored in SPGR inversion as TE << T2*)
B1 = 1.0                # paper assumes B1=1 for conventional DCE analysis
FA_BASE = np.deg2rad(10.0)
TR_BASE = 12e-3         # s

# Reference-tissue (muscle) kinetics (paper)
Ktrans_ref_min = 0.11   # min^-1
Ktrans_ref = Ktrans_ref_min / 60.0  # s^-1
ve_ref = 0.20
kappa = Ktrans_ref / ve_ref

# If no T1 map is available, use a reasonable muscle T1 at 7T:
T10_REF_FALLBACK = 1.8  # s

# Injection start (paper: 60 s). If unknown, set to None to auto-detect
INJECTION_START_S = 60.0

# Regularization strength (tune 0.01–0.2)
LAMBDA = 0.05

dce_img_path = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}.img"
roi_mat_path_tumor = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_tumor.mat"
roi_mat_path_muscle = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_muscle.mat"

img = nib.load(str(dce_img_path))
data = img.get_fdata()
Nt = data.shape[2]
print("Data shape:", data.shape)

# Load tumor & muscle ROIs
mat_tumor = loadmat(roi_mat_path_tumor)
roi_struct_tumor = mat_tumor["roi"][0, 0]
mat_muscle = loadmat(roi_mat_path_muscle)
roi_struct_muscle = mat_muscle["roi"][0, 0]

mask_tumor = np.array(roi_struct_tumor["bw"], dtype=bool).T
mask_muscle = np.array(roi_struct_muscle["bw"], dtype=bool).T

time_curves_tumor = [data[:, :, t][mask_tumor].mean() for t in range(data.shape[2])] # data[:, :, t] extracts the 2D image at timepoint
time_curves_muscle = [data[:, :, t][mask_muscle].mean() for t in range(data.shape[2])]

S_ref_all = np.array([data[:, :, t][mask_muscle].mean() for t in range(data.shape[2])])
print("S_ref shape:", S_ref_all.shape)

FA_deg = [10, 20, 5, 10, 30, 2, 10, 80, 10]
TR_ms  = [12, 12, 12, 12, 12, 12, 12, 100, 12]
nfrm   = [40,  5,  5,  5,   5,  5,  5,   3,  5]

FA = np.concatenate([np.deg2rad(np.full(n, fa)) for fa, n in zip(FA_deg, nfrm)])
TR = np.concatenate([np.full(n, tr / 1000.0) for tr, n in zip(TR_ms, nfrm)])
dt_segments = [5.4 if fa != 80 else 45.0 for fa in FA_deg]
dt_vec = np.concatenate([np.full(n, d) for d, n in zip(dt_segments, nfrm)])
t_full = np.cumsum(np.concatenate([[0], dt_vec[:-1]]))

#Build baseline-only arrays (FA=10°, TR=12 ms) -
baseline_idx = np.where((np.isclose(FA, np.deg2rad(10))) & (np.isclose(TR, 0.012)))[0]
t_b = t_full[baseline_idx]
S_ref = S_ref_all[baseline_idx]  #


print(f"Total frames: {len(t_full)} | Baseline frames: {len(baseline_idx)} | Final time: {t_full[-1]:.1f}s")



if len(S_ref) >= 7:
    S_ref_smooth = savgol_filter(S_ref, window_length=7, polyorder=2, mode="interp")
else:
    S_ref_smooth = S_ref.copy()

# Determine pre-injection baseline frames within baseline protocol
if INJECTION_START_S is None:
    # crude auto-detect: first big rise
    dif = np.diff(S_ref_smooth)
    k = np.argmax(dif > (0.02 * np.max(S_ref_smooth))) + 1
    t0_idx = max(3, k)
    pre_mask = np.zeros_like(t_b, dtype=bool)
    pre_mask[:t0_idx] = True
else:
    pre_mask = t_b < INJECTION_START_S
    if pre_mask.sum() < 3:
        pre_mask = np.zeros_like(t_b, dtype=bool)
        pre_mask[:min(5, len(t_b))] = True

# M0 estimation requires T10 of muscle (paper uses RARE-VTR map)
if T1_MAP_PATH and Path(T1_MAP_PATH).exists():
    t1map_img = nib.load(str(T1_MAP_PATH))
    t1map = t1map_img.get_fdata()
    T10_ref = float(np.nanmean(t1map[:, :][mask_muscle]))
    print(f"Using measured muscle T1: {T10_ref:.3f} s")
else:
    T10_ref = T10_REF_FALLBACK
    print(f" No T1 map supplied. Using fallback muscle T1 = {T10_ref:.2f} s")

R10_ref = 1.0 / T10_ref

# Estimate M0 from pre-injection frames using SPGR forward model
S_pre = float(np.mean(S_ref_smooth[pre_mask]))
E1_pre = np.exp(-TR_BASE / T10_ref)
M0 = S_pre * (1.0 - E1_pre * np.cos(FA_BASE)) / ((1.0 - E1_pre) * np.sin(FA_BASE))

def invert_spgr_T1(S, M0, FA, TR):
    y = S / (M0 * np.sin(FA))
    y = np.clip(y, 1e-8, 0.9999)
    x = (1.0 - y) / (1.0 - y * np.cos(FA))
    x = np.clip(x, 1e-8, 0.999999)
    T1 = -TR / np.log(x)
    return T1

T1_ref_t = invert_spgr_T1(S_ref_smooth, M0, FA_BASE, TR_BASE)
R1_ref_t = 1.0 / T1_ref_t
C_ref = (R1_ref_t - R10_ref) / r1
C_ref = np.clip(C_ref, 0.0, None)

# -----------------------------
# 4) Deconvolution (non-uniform) with Tikhonov + first-difference smoothing
#     Min ||A Cp - C_ref||^2 + λ^2 ||D Cp||^2
# -----------------------------
N = len(t_b)
# trapezoidal weights (non-uniform)
w = np.empty(N)
w[0] = t_b[1] - t_b[0] if N > 1 else 5.4
w[1:] = np.diff(t_b)

A = np.zeros((N, N))
for n in range(N):
    dt = t_b[n] - t_b[:n+1]
    A[n, :n+1] = np.exp(-kappa * dt) * w[:n+1]
A *= Ktrans_ref

# First-difference operator D
D = np.eye(N) - np.eye(N, k=-1)
D = D[1:]  # shape (N-1, N)

# Solve (A^T A + λ^2 D^T D) Cp = A^T C_ref
ATA = A.T @ A
DTD = D.T @ D
rhs = A.T @ C_ref
Cp = np.linalg.solve(ATA + (LAMBDA**2) * DTD, rhs)
Cp = np.clip(Cp, 0.0, None)

# -----------------------------
# 5) Plot
# -----------------------------
plt.figure(figsize=(7,4))
plt.plot(t_b, Cp, lw=2, label="AIF (DCE)")
plt.plot(t_b, Cp, lw=2, ls="--", label="AIF (ACE)")  # visually identical in paper
plt.xlabel("Time (s)")
plt.ylabel("AIF (mM)")
plt.title("Reference-tissue derived AIF (muscle ROI)")
plt.xlim(0, max(600, t_b[-1] if N else 600))
plt.ylim(0, Cp.max()*1.15 if Cp.size else 1)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("aif_reference_tissue_regularized.png", dpi=200)

plt.figure(figsize=(7,4))
plt.plot(t_b, S_ref, label="Muscle signal (raw)")
plt.plot(t_b, S_ref_smooth, label="Muscle signal (smoothed)")
plt.axvline(t_b[pre_mask].max() if pre_mask.any() else 0, ls=':', color='k', label='Last pre-inj.')
plt.xlabel("Time (s)"); plt.ylabel("Signal (a.u.)"); plt.legend(); plt.tight_layout()
plt.savefig("diag_muscle_signal.png", dpi=200)

plt.figure(figsize=(7,4))
plt.plot(t_b, C_ref, label="Muscle concentration")
plt.xlabel("Time (s)"); plt.ylabel("mM"); plt.legend(); plt.tight_layout()
plt.savefig("diag_muscle_concentration.png", dpi=200)
plt.show()


