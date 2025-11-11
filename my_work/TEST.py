
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

S_ref_all = np.array([data[:, :, t][mask_muscle].mean() for t in range(data.shape[2])]) #reference tissue signal
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

# M0 estimation requires T10 of muscle
if T1_MAP_PATH and Path(T1_MAP_PATH).exists():
    t1map_img = nib.load(str(T1_MAP_PATH))
    t1map = t1map_img.get_fdata()
    T10_ref = float(np.nanmean(t1map[:, :][mask_muscle])) #extracts all voxels in the muscle mask and takes their mean
    print(f"Using measured muscle T1: {T10_ref:.3f} s")
else:
    T10_ref = T10_REF_FALLBACK
    print(f" No T1 map supplied. Using fallback muscle T1 = {T10_ref:.2f} s")

R10_ref = 1.0 / T10_ref

# Estimate M0 from pre-injection frames using SPGR forward model
# Instead of averaging all pre_mask frames, fit M0 from a linearized SPGR
S_pre = S_ref_smooth[pre_mask]
n = S_pre.size
x = S_pre * np.sin(FA_BASE) / (1 - np.exp(-TR_BASE/T10_ref) * np.cos(FA_BASE))
E1_pre = np.exp(-TR_BASE / T10_ref)
M0 = np.mean(S_ref_smooth[pre_mask]) * (1 - E1_pre * np.cos(FA_BASE)) / ((1 - E1_pre) * np.sin(FA_BASE))
M0 = M0 * 0.8   # reduce by ~20% if negative C_ref (empirical correction)

print(f"Refitted M0 = {M0:.3f}")


def invert_spgr_T1(S, M0, FA, TR):
    y = S / (M0 * np.sin(FA))
    y = np.clip(y, 1e-8, 0.9999)
    x = (1.0 - y) / (1.0 - y * np.cos(FA))
    x = np.clip(x, 1e-8, 0.999999)
    T1 = -TR / np.log(x)
    return T1

T1_ref_t = invert_spgr_T1(S_ref_smooth, M0, FA_BASE, TR_BASE)
print("T1 range:", T1_ref_t.min(), T1_ref_t.max())
print("Baseline T1:", T1_ref_t[pre_mask].mean())

R1_ref_t = 1.0 / T1_ref_t
C_ref = (R1_ref_t - R10_ref) / r1
#C_ref = np.clip(C_ref, 0.0, None)
from scipy.signal import savgol_filter
if len(C_ref) >= 9:
    C_ref = savgol_filter(C_ref, 9, 3)


C_ref_bc = C_ref - np.mean(C_ref[pre_mask])


# -----------------------------
# 4) Deconvolution (non-uniform) with Tikhonov + first-difference smoothing
#     Cp = argmin ||A Cp - C_ref||^2 + λ^2 ||D Cp||^2,  A = Ktrans_ref * (K ∘ Δt) + vp_ref * I
# -----------------------------
# -----------------------------
# (A) Baseline centering (only mean, no over-correction)
# -----------------------------
# High-pass via polynomial detrend
p = np.polyfit(t_b[pre_mask], C_ref[pre_mask], 2)
C_ref_bc = C_ref - np.polyval(p, t_b)

# -----------------------------
# (B) Build non-uniform trapezoid integration weights (seconds)
# -----------------------------
N = len(t_b)
dt = np.diff(t_b)
Delta = np.zeros(N, dtype=float)
if N == 1:
    Delta[0] = 5.4
else:
    Delta[0]    = dt[0] / 2.0
    Delta[1:-1] = (dt[:-1] + dt[1:]) / 2.0
    Delta[-1]   = dt[-1] / 2.0

# -----------------------------
# (C) Convolution matrix for the reference tissue (extended Tofts, vp_ref ≈ 0)
# C_ref ≈ Ktrans_ref * ∫_0^t Cp(τ) * exp(-kappa*(t-τ)) dτ  + vp_ref * Cp(t)
# Discretize with trapezoid weights Delta
# -----------------------------
K = np.zeros((N, N), dtype=float)  # lower triangular kernel
for n in range(N):
    tj = t_b[:n+1]
    K[n, :n+1] = np.exp(-kappa * (t_b[n] - tj))
A = Ktrans_ref * (K * Delta)       # column-wise scale by Δt_j
vp_ref = 0.0                        # match the paper's reference-tissue AIF
if vp_ref != 0.0:
    A = A + vp_ref * np.eye(N)

# -----------------------------
# (D) Regularized nonnegative least squares
# Minimize: ||A Cp - C_ref_bc||^2 + λ^2 ||D Cp||^2,  subject to Cp >= 0
# Solve via bounded least-squares on the *augmented* system:
#   [A        ] Cp ≈ [C_ref_bc]
#   [λ D      ]       [0       ]
# -----------------------------
from scipy.optimize import least_squares
LAMBDA = 0.12
# first-difference regularizer
D = np.zeros((N-1, N), dtype=float)
for i in range(N-1):
    D[i, i]   = -1.0
    D[i, i+1] =  1.0

lam = max(1e-3, float(LAMBDA))  # 0.05 is a good start; try 0.05–0.15
sqrt_lam = np.sqrt(lam)

def residual(c):
    r_data = A @ c - C_ref_bc
    r_reg  = sqrt_lam * (D @ c)
    return np.hstack([r_data, r_reg])

Cp0 = np.maximum(0.0, C_ref_bc / (Ktrans_ref + 1e-8))  # simple nonnegative init
res = least_squares(residual, Cp0, bounds=(0, np.inf), method="trf", max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
Cp = res.x

# -----------------------------
# (E) Sanity check: does A @ Cp reproduce the muscle curve?
# -----------------------------
C_ref_fit = A @ Cp

print(f"||data residual||/||data|| = {np.linalg.norm(C_ref_fit - C_ref_bc)/max(1e-8, np.linalg.norm(C_ref_bc)):.3f}")

# -----------------------------

# (F) Plots: (1) muscle concentration vs. fit, (2) AIF
# -----------------------------
plt.figure(figsize=(7.2, 4.0))
plt.plot(t_b/60, C_ref_bc, lw=1.8, label="Muscle C_ref (baseline-corrected)")
plt.plot(t_b/60, C_ref_fit, lw=1.8, ls="--", label="Model fit (A·Cp)")
plt.xlabel("Time (min)"); plt.ylabel("Concentration (mM)")
plt.title("Reference tissue fit check")
plt.grid(alpha=0.3); plt.legend(frameon=False); plt.tight_layout()
plt.show()

plt.figure(figsize=(7.2, 4.0))
plt.plot(t_b/60, Cp, lw=2.2, label="AIF (DCE, reference-tissue)")
plt.xlabel("Time (min)"); plt.ylabel("Concentration (mM)")
plt.title("Reference-tissue AIF (muscle ROI)")
plt.grid(alpha=0.3); plt.legend(frameon=False); plt.tight_layout()
plt.show()
