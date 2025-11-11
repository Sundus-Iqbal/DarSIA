import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from scipy.io import loadmat
from scipy.signal import savgol_filter
from skimage.draw import polygon

# ------------------ config ------------------
DATA_ROOT = Path("/home/siq/Data_ACEMRI")
MOUSE = "M01_Post"
SLICE = "Slice5fid"

T10_REF = 1.50             # s (muscle at 7T)
R1      = 3.5              # s^-1 mM^-1
KTRANS_REF_MIN = 0.25      # min^-1 (muscle)
VE_REF         = 0.10
BOLUS_DELAY_S  = 5.0       # s (0 to disable)

X_LIM_S = (0, 600)
Y_LIM_M = (0, 3.0)

SMOOTH_SEC_SIGNAL = 10     # SavGol window (on native time grid) BEFORE T1 inversion
SMOOTH_SEC_CREF   = 12     # SavGol window on uniform grid BEFORE derivative
POLYORDER         = 3
DCDT_CLIP = (-0.02, 0.02)  # mM/s

# ------------------ helpers ------------------
def largest_odd_leq(n): return n if n % 2 else max(1, n-1)
def odd_window_for_seconds(n_samples, seconds, t):
    if n_samples < 3: return 1
    dt = float(np.median(np.diff(t)))
    w  = int(np.floor(seconds / max(dt, 1e-9)))
    w  = max(7, w|1)                          # at least 7, odd
    return min(largest_odd_leq(w), n_samples - (1 - (n_samples % 2)))

def load_roi_polygon_mask(mat_path, target_shape):
    m = loadmat(mat_path)
    r = m['roi'][0,0]
    xi = np.squeeze(r['xi']).astype(float) - 1
    yi = np.squeeze(r['yi']).astype(float) - 1
    base = np.zeros(target_shape, dtype=bool)
    rr, cc = polygon(yi, xi, base.shape)
    base[rr, cc] = True
    for msk in [base, base.T, np.flipud(base), np.fliplr(base),
                np.flipud(base.T), np.fliplr(base.T),
                np.flipud(np.fliplr(base)), np.flipud(np.fliplr(base.T))]:
        if msk.shape == target_shape: return msk
    raise ValueError("ROI mask shape mismatch")

def estimate_M0_ls(S, FA_deg, TR_s, T10):
    # LS over baseline frames (first 8 or all pre-injection if you know them)
    nb = min(12, len(S))
    a  = np.deg2rad(FA_deg[:nb]); TR = TR_s[:nb]
    E10 = np.exp(-TR / T10)
    x = np.sin(a) * (1 - E10) / (1 - np.cos(a) * E10 + 1e-12)
    M0 = np.maximum(np.dot(S[:nb], x) / np.dot(x, x + 1e-12), 1e-6)
    return float(M0)

def invert_T1_SPGR(S, alpha_deg, TR_s, M0, T1_init):
    # Newton solve per frame (robust, bounded)
    a = np.deg2rad(alpha_deg)
    S = np.clip(S, 1e-10, None)
    T1 = np.full_like(S, T1_init, dtype=float)
    for _ in range(40):
        E1 = np.exp(-TR_s / np.clip(T1, 1e-6, None))
        num = M0 * np.sin(a) * (1 - E1)
        den = (1 - np.cos(a) * E1)
        f   = num / (den + 1e-12) - S
        dE  = (TR_s / (T1**2)) * E1
        dnum = -M0 * np.sin(a) * dE
        dden =  np.cos(a) * dE
        df = (dnum*den - num*dden) / ((den + 1e-12)**2)
        step = f / (df + 1e-12)
        T1 = np.clip(T1 - step, 0.05, 6.0)
        if np.nanmax(np.abs(step)) < 1e-5: break
    return T1

def segment_arrays(values, seg_starts, seg_ends):
    return [values[s:e] for s, e in zip(seg_starts, seg_ends)]

def per_segment_gain_normalize(S, t, seg_starts, seg_ends):
    # Align segment medians over the first few seconds to reduce receive-gain jumps
    chunks = segment_arrays(S, seg_starts, seg_ends)
    t_chunks = segment_arrays(t, seg_starts, seg_ends)
    anchors = []
    for s, tc in zip(chunks, t_chunks):
        k = min(5, len(s))
        anchors.append(np.median(s[:k]))
    ref = anchors[0] if anchors[0] > 0 else 1.0
    scales = [ref / a if a > 0 else 1.0 for a in anchors]
    # apply
    S_corr = np.empty_like(S)
    for i,(s,e) in enumerate(zip(seg_starts, seg_ends)):
        S_corr[s:e] = S[s:e] * scales[i]
    return S_corr, scales

# ------------------ load data ------------------
img_path = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}.img"
roi_mus  = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_muscle.mat"
roi_tum  = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_tumor.mat"

img = nib.load(str(img_path))
data = img.get_fdata()                  # (X,Y,T)
X, Y, T = data.shape

mask_mus = load_roi_polygon_mask(roi_mus, data.shape[:2])
S_ref_mean = np.nanmean(data[mask_mus, :], axis=0)  # (T,)

# ------------------ ACE timing ------------------
FA_DEG = np.array([10,20, 5,10,30, 2,10,80,10], dtype=float)
TR_MS  = np.array([12,12,12,12,12,12,12,100,12], dtype=float)
NFR    = np.array([40, 5, 5, 5, 5, 5, 5,  3, 5], dtype=int)
DT_S   = np.where(FA_DEG < 80, 5.4, 45.0)

FA_pf  = np.concatenate([np.full(n, FA_DEG[i]) for i,n in enumerate(NFR)])
TR_pf  = np.concatenate([np.full(n, TR_MS[i])  for i,n in enumerate(NFR)]) / 1000.0
dt_pf  = np.concatenate([np.full(n, DT_S[i])   for i,n in enumerate(NFR)])
assert len(FA_pf) == len(TR_pf) == len(dt_pf) == T

t_s = np.cumsum(dt_pf); t_s -= t_s[0]
seg_ends   = np.cumsum(NFR)
seg_starts = np.concatenate(([0], seg_ends[:-1]))
baseline_segments = [0,3,6,8]
dce_idx = np.concatenate([np.arange(seg_starts[i], seg_ends[i]) for i in baseline_segments])

# ------------------ (1) smooth signal per native timeline + gain normalize ------------------
# per-segment gain normalization (mitigates receiver gain changes when FA/TR switch)
S_ref_gn, seg_scales = per_segment_gain_normalize(S_ref_mean, t_s, seg_starts, seg_ends)

# SavGol on native (irregular) timeline
win_native = odd_window_for_seconds(len(S_ref_gn), SMOOTH_SEC_SIGNAL, t_s)
S_ref_sm = savgol_filter(S_ref_gn, window_length=win_native, polyorder=POLYORDER, mode='interp')

# ------------------ (2) estimate M0 from baseline and invert to T1 -> Cref ------------------
M0_est = estimate_M0_ls(S_ref_sm, FA_pf, TR_pf, T10_REF)

def signal_to_Cref(S, FA, TR, M0, T10, r1):
    T1_t = invert_T1_SPGR(S, FA, TR, M0, T1_init=T10)
    C = (1.0 / np.clip(T1_t, 1e-6, None) - 1.0 / T10) / r1
    return np.clip(C, 0.0, None)

Cref_native = signal_to_Cref(S_ref_sm, FA_pf, TR_pf, M0_est, T10_REF, R1)

# ------------------ (3) resample to uniform time, smooth, derivative ------------------
t_u = np.arange(X_LIM_S[0], min(X_LIM_S[1], int(np.floor(t_s[-1]))) + 1, 1.0)
Cref_u = np.interp(t_u, t_s, Cref_native)

# optional bolus delay (shift ref to earlier to approximate arterial timing)
if BOLUS_DELAY_S != 0.0:
    Cref_u = np.interp(t_u, t_u + BOLUS_DELAY_S, Cref_u, left=Cref_u[0], right=Cref_u[-1])

win_cref = odd_window_for_seconds(len(Cref_u), SMOOTH_SEC_CREF, t_u)
Cref_s   = savgol_filter(Cref_u, window_length=win_cref, polyorder=POLYORDER, mode='interp')

dCref_dt = np.gradient(Cref_s, t_u)
if DCDT_CLIP is not None:
    lo, hi = DCDT_CLIP
    dCref_dt = np.clip(dCref_dt, lo, hi)

# ------------------ (4) construct Cp(t) and light post-smoothing ------------------
Kt_ref_s = KTRANS_REF_MIN / 60.0
Kep_ref  = Kt_ref_s / (VE_REF + 1e-12)
Cp = (dCref_dt + Kep_ref * Cref_s) / (Kt_ref_s + 1e-12)
Cp = np.clip(Cp, 0.0, None)

Cp = savgol_filter(Cp, window_length=win_cref, polyorder=POLYORDER, mode='interp')

# ------------------ diagnostics and plot ------------------
plt.figure(figsize=(8, 3.2))
plt.plot(t_s, S_ref_mean,  ':', label='muscle signal (raw)')
plt.plot(t_s, S_ref_gn,    '--', label='gain-normalized')
plt.plot(t_s, S_ref_sm,    '-',  label='signal smoothed')
plt.legend(); plt.xlabel('Time (s)'); plt.ylabel('Signal'); plt.title('Reference signal diagnostics'); plt.tight_layout()

plt.figure(figsize=(8, 3.2))
plt.plot(t_u, Cref_u, ':', label='C_ref (uniform)')
plt.plot(t_u, Cref_s, '-', label='C_ref (smoothed)')
plt.plot(t_u, dCref_dt, '--', label='dC_ref/dt (clipped)')
plt.legend(); plt.xlabel('Time (s)'); plt.ylabel('mM / mM/s'); plt.title('Concentration + derivative'); plt.tight_layout()

plt.figure(figsize=(8, 3.6))
plt.plot(t_u, Cp, '-', label='AIF via reference tissue')
plt.xlim(*X_LIM_S); plt.ylim(*Y_LIM_M)
plt.xlabel('Time (s)'); plt.ylabel('AIF $C_p(t)$ [mM]')
plt.title('Stabilized AIF')
plt.legend(); plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import nibabel as nib
# from scipy.io import loadmat
# from scipy.signal import savgol_filter
# from skimage.draw import polygon
#
# # -----------------------------
# # Config / paths
# # -----------------------------
# DATA_ROOT = Path("/home/siq/Data_ACEMRI")
# MOUSE = "M01_Post"
# SLICE = "Slice5fid"
#
# dce_img_path = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}.img"
# roi_mat_path_tumor  = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_tumor.mat"
# roi_mat_path_muscle = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_muscle.mat"
#
# # -----------------------------
# # Acquisition / model constants
# # (tweakable; these drive the AIF shape)
# # -----------------------------
# T10_REF = 1.50     # s, baseline T1 of muscle at 7T (1.5s typical; 1.8s can flatten curves)
# R1      = 3.5      # s^-1 mM^-1, gadolinium relaxivity (sequence- & field-dependent)
#
# KTRANS_REF_MIN = 0.25  # min^-1, muscle reference Ktrans (try 0.20–0.35 min^-1)
# VE_REF         = 0.10  # unitless, muscle ve (0.08–0.15 typical)
#
# BOLUS_DELAY_S  = 5.0   # s, optional shift (muscle lags artery). Set 0.0 to disable.
#
# # AIF plot limits
# X_LIM_S = (0, 600)    # seconds
# Y_LIM_M = (0, 3.0)    # mM
#
# # Smoothing parameters (in seconds of data on the uniform grid)
# SMOOTH_WIN_SEC = 15      # ~10–20 s works well
# POLYORDER      = 3
#
# # Derivative clipping to tame noise (mM/s). Set to None to disable.
# DCDT_CLIP = (-0.02, 0.02)
#
# # Save figures
# SAVE_DIR = Path("./_aif_out")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)
#
# # -----------------------------
# # Utilities
# # -----------------------------
# def largest_odd_leq(n: int) -> int:
#     n = int(n)
#     return n if n % 2 else max(1, n - 1)
#
# def odd_window_for_seconds(n_samples: int, seconds: float, dt: float, min_win: int = 9) -> int:
#     """Return an odd Savitzky-Golay window length approximating 'seconds' at sampling dt."""
#     if n_samples <= 2:
#         return 1
#     approx = int(np.floor(seconds / max(dt, 1e-9)))
#     approx = max(min_win, approx | 1)  # make odd, at least min_win
#     return min(largest_odd_leq(approx), n_samples - (1 - (n_samples % 2)))  # must be <= n and odd
#
# def load_roi_polygon_mask(mat_path, target_shape):
#     """
#     Load MATLAB roi struct with fields ('xi','yi') and fill polygon to boolean mask.
#     Handles MATLAB 1-based indexing and common orientation cases.
#     """
#     m = loadmat(mat_path)
#     if 'roi' not in m:
#         raise KeyError(f"'roi' not found in {mat_path}")
#     r = m['roi'][0, 0]
#     if 'xi' not in r.dtype.names or 'yi' not in r.dtype.names:
#         raise KeyError(f"'xi'/'yi' not found in roi struct of {mat_path}")
#
#     xi = np.squeeze(r['xi']).astype(float) - 1  # MATLAB -> Python
#     yi = np.squeeze(r['yi']).astype(float) - 1
#
#     base = np.zeros(target_shape, dtype=bool)
#     rr, cc = polygon(yi, xi, base.shape)  # rr: y, cc: x
#     base[rr, cc] = True
#
#     candidates = [
#         base, base.T, np.flipud(base), np.fliplr(base),
#         np.flipud(base.T), np.fliplr(base.T),
#         np.flipud(np.fliplr(base)), np.flipud(np.fliplr(base.T)),
#     ]
#     for msk in candidates:
#         if msk.shape == target_shape:
#             return msk
#     raise ValueError(f"ROI mask from {mat_path} could not be coerced to shape {target_shape}")
#
# def estimate_M0_from_baseline(S, FA_deg_series, TR_s_series, T10_ref, n_baseline=8):
#     """Median SPGR linearization over first n_baseline frames."""
#     nb = min(n_baseline, len(S))
#     alpha = np.deg2rad(FA_deg_series[:nb])
#     TRs   = TR_s_series[:nb]
#     E10   = np.exp(-TRs / T10_ref)
#     denom = np.sin(alpha) * (1.0 - E10) / (1.0 - np.cos(alpha) * E10 + 1e-12)
#     M0 = np.median(S[:nb] / np.maximum(denom, 1e-12))
#     return float(M0)
#
# def invert_T1_from_SPGR(S, alpha_deg, TR_s, M0, T10_ref, iters=40):
#     """Newton-Raphson inversion for SPGR T1."""
#     alpha = np.deg2rad(alpha_deg)
#     S = np.clip(S, 1e-12, None)
#     T1 = np.full_like(S, T10_ref, dtype=float)
#     for _ in range(iters):
#         E1 = np.exp(-TR_s / np.clip(T1, 1e-6, None))
#         num   = M0 * np.sin(alpha) * (1 - E1)
#         den   = (1 - np.cos(alpha) * E1)
#         f     = num / (den + 1e-12) - S
#         dE1dT1 = (TR_s / (T1**2)) * E1
#         d_num  = -M0 * np.sin(alpha) * dE1dT1
#         d_den  =  np.cos(alpha) * dE1dT1
#         df = (d_num * den - num * d_den) / ((den + 1e-12)**2)
#         step = f / (df + 1e-12)
#         T1 = np.clip(T1 - step, 0.05, 6.0)  # 50 ms .. 6 s
#         if np.nanmax(np.abs(step)) < 1e-5:
#             break
#     return T1
#
# def signal_to_concentration(S_mean, FA_deg_series, TR_s_series, M0, T10_ref, r1):
#     T1_t = invert_T1_from_SPGR(S_mean, FA_deg_series, TR_s_series, M0, T10_ref)
#     C_t  = (1.0 / np.clip(T1_t, 1e-6, None) - 1.0 / T10_ref) / r1
#     return np.clip(C_t, 0.0, None)
#
# def smooth_and_derivative(t, y, win_sec, polyorder=3, dcdt_clip=None):
#     """Savitzky–Golay smooth (on uniform or nonuniform t) and robust derivative."""
#     t = np.asarray(t)
#     y = np.asarray(y)
#     dt_median = np.median(np.diff(t))
#     win = odd_window_for_seconds(len(y), win_sec, dt_median, min_win=9)
#     y_s = savgol_filter(y, window_length=win, polyorder=polyorder, mode='interp')
#     # robust derivative from the smoothed curve
#     dy_dt = np.gradient(y_s, t)
#     if dcdt_clip is not None:
#         lo, hi = dcdt_clip
#         dy_dt = np.clip(dy_dt, lo, hi)
#     return y_s, dy_dt, win
#
# def compute_reference_aif(t, Cref, Ktrans_ref_min, ve_ref,
#                           smooth_win_sec=15, polyorder=3,
#                           dcdt_clip=(-0.02, 0.02),
#                           bolus_delay_s=0.0):
#     """
#     Stable reference-tissue AIF:
#       Cp = [ dCref/dt + (Ktrans_ref/ve_ref)*Cref ] / Ktrans_ref
#     with smoothing before derivative, derivative clipping, and optional delay.
#     """
#     t = np.asarray(t).astype(float)
#     Cref = np.asarray(Cref).astype(float)
#
#     # Optional bolus delay (shift reference tissue to the left: earlier arterial input)
#     if bolus_delay_s != 0.0:
#         Cref = np.interp(t, t + bolus_delay_s, Cref, left=Cref[0], right=Cref[-1])
#
#     # Smooth + derivative
#     Cref_s, dCref_dt, used_win = smooth_and_derivative(
#         t, Cref, win_sec=smooth_win_sec, polyorder=polyorder, dcdt_clip=dcdt_clip
#     )
#
#     # Units: convert Ktrans to s^-1
#     Kt_ref_s = Ktrans_ref_min / 60.0
#     Kep_ref  = Kt_ref_s / (ve_ref + 1e-12)
#
#     Cp = (dCref_dt + Kep_ref * Cref_s) / (Kt_ref_s + 1e-12)
#     Cp = np.clip(Cp, 0.0, None)
#
#     # Light display smoothing (same window) to remove tiny wiggles
#     dt_median = np.median(np.diff(t))
#     win = odd_window_for_seconds(len(Cp), smooth_win_sec, dt_median, min_win=9)
#     Cp_s = savgol_filter(Cp, window_length=win, polyorder=polyorder, mode='interp')
#
#     info = {
#         "Ktrans_ref_s_inv": Kt_ref_s,
#         "Kep_ref": Kep_ref,
#         "smooth_window_samples": used_win,
#         "dt_median": float(dt_median),
#     }
#     return Cp_s, Cref_s, dCref_dt, info
#
# # -----------------------------
# # Load DCE 4D (x,y,t) and ROIs
# # -----------------------------
# img = nib.load(str(dce_img_path))
# data = img.get_fdata()
# assert data.ndim == 3, f"Expected 3D+time DCE array, got shape {data.shape}"
# X, Y, T = data.shape
# print("DCE image shape:", data.shape)
#
# mask_muscle = load_roi_polygon_mask(roi_mat_path_muscle, data.shape[:2])
# mask_tumor  = load_roi_polygon_mask(roi_mat_path_tumor,  data.shape[:2])
# print("Muscle voxels:", np.sum(mask_muscle), "Tumor voxels:", np.sum(mask_tumor))
#
# # -----------------------------
# # ACE protocol timeline
# # -----------------------------
# FA_DEG = np.array([10, 20,  5, 10, 30,  2, 10, 80, 10], dtype=float)   # degrees
# TR_MS  = np.array([12, 12, 12, 12, 12, 12, 12, 100, 12], dtype=float)  # ms
# NFR    = np.array([40,  5,  5,  5,  5,  5,  5,   3,  5], dtype=int)    # frames/segment
# DT_S   = np.where(FA_DEG < 80, 5.4, 45.0)                              # s/frame per segment
#
# FA_pf  = np.concatenate([np.full(n, FA_DEG[i]) for i, n in enumerate(NFR)])
# TR_pf  = np.concatenate([np.full(n, TR_MS[i])  for i, n in enumerate(NFR)]) / 1000.0
# dt_pf  = np.concatenate([np.full(n, DT_S[i])   for i, n in enumerate(NFR)])
# assert len(FA_pf) == len(TR_pf) == len(dt_pf) == T == 78, "Per-frame series length mismatch"
#
# # Absolute time stamps (start at 0)
# t_s = np.cumsum(dt_pf); t_s = t_s - t_s[0]
#
# # DCE-only subset indices (baseline segments: 1,4,7,9 => indices 0,3,6,8)
# seg_ends   = np.cumsum(NFR)
# seg_starts = np.concatenate(([0], seg_ends[:-1]))
# baseline_segments = [0, 3, 6, 8]
# dce_idx = np.concatenate([np.arange(seg_starts[i], seg_ends[i]) for i in baseline_segments])
#
# # -----------------------------
# # Build reference (muscle) signal
# # -----------------------------
# S_ref = data[mask_muscle, :]               # (Nvox, T)
# S_ref_mean = np.nanmean(S_ref, axis=0)     # (T,)
#
# # -----------------------------
# # SPGR inversion: signal → concentration
# # -----------------------------
# M0_est = estimate_M0_from_baseline(S_ref_mean, FA_pf, TR_pf, T10_REF, n_baseline=8)
#
# # Full ACE timeline
# Cref_ACE_native = signal_to_concentration(S_ref_mean, FA_pf, TR_pf, M0_est, T10_REF, R1)
#
# # DCE-only frames (subset)
# Cref_DCE_native = signal_to_concentration(
#     S_ref_mean[dce_idx], FA_pf[dce_idx], TR_pf[dce_idx], M0_est, T10_REF, R1
# )
# t_s_DCE = t_s[dce_idx]
#
# # -----------------------------
# # Interpolate to uniform 1-s grid
# # -----------------------------
# t_uniform = np.arange(X_LIM_S[0], min(X_LIM_S[1], int(np.floor(t_s[-1]))) + 1, 1.0)
# Cref_ACE_u = np.interp(t_uniform, t_s,     np.clip(Cref_ACE_native, 0, None))
# Cref_DCE_u = np.interp(t_uniform, t_s_DCE, np.clip(Cref_DCE_native, 0, None))
#
# # -----------------------------
# # Compute AIFs (ACE and DCE)
# # -----------------------------
# Cp_ACE,  Cref_ACE_s,  dCref_ACE_dt,  info_ACE  = compute_reference_aif(
#     t_uniform, Cref_ACE_u,
#     Ktrans_ref_min=KTRANS_REF_MIN, ve_ref=VE_REF,
#     smooth_win_sec=SMOOTH_WIN_SEC, polyorder=POLYORDER,
#     dcdt_clip=DCDT_CLIP, bolus_delay_s=BOLUS_DELAY_S
# )
# Cp_DCE,  Cref_DCE_s,  dCref_DCE_dt,  info_DCE  = compute_reference_aif(
#     t_uniform, Cref_DCE_u,
#     Ktrans_ref_min=KTRANS_REF_MIN, ve_ref=VE_REF,
#     smooth_win_sec=SMOOTH_WIN_SEC, polyorder=POLYORDER,
#     dcdt_clip=DCDT_CLIP, bolus_delay_s=BOLUS_DELAY_S
# )
#
# print("ACE info:", info_ACE)
# print("DCE info:", info_DCE)
#
# # -----------------------------
# # Diagnostics: muscle concentration & derivative
# # -----------------------------
# plt.figure(figsize=(8, 4))
# plt.plot(t_uniform, Cref_ACE_u,  ':',  label='ACE muscle C_ref (raw interp)')
# plt.plot(t_uniform, Cref_ACE_s,  '-',  label='ACE muscle C_ref (smoothed)')
# plt.plot(t_uniform, dCref_ACE_dt, '--', label='ACE dC_ref/dt (smoothed grad)')
# plt.xlabel('Time (s)'); plt.ylabel('Muscle C_ref / dC_ref/dt')
# plt.title('Reference Tissue Diagnostics (ACE)')
# plt.legend(); plt.tight_layout()
# plt.savefig(SAVE_DIR / "diagnostic_muscle_ace.png", dpi=160)
#
# plt.figure(figsize=(8, 4))
# plt.plot(t_uniform, Cref_DCE_u,  ':',  label='DCE muscle C_ref (raw interp)')
# plt.plot(t_uniform, Cref_DCE_s,  '-',  label='DCE muscle C_ref (smoothed)')
# plt.plot(t_uniform, dCref_DCE_dt, '--', label='DCE dC_ref/dt (smoothed grad)')
# plt.xlabel('Time (s)'); plt.ylabel('Muscle C_ref / dC_ref/dt')
# plt.title('Reference Tissue Diagnostics (DCE)')
# plt.legend(); plt.tight_layout()
# plt.savefig(SAVE_DIR / "diagnostic_muscle_dce.png", dpi=160)
#
# # -----------------------------
# # Final AIF plot
# # -----------------------------
# plt.figure(figsize=(8, 4))
# plt.plot(t_uniform, Cp_DCE, '-',  label='DCE-MRI (reference-tissue AIF)')
# plt.plot(t_uniform, Cp_ACE, '--', label='ACE-MRI (reference-tissue AIF)')
# plt.xlabel('Time (s)')
# plt.ylabel('AIF  $C_p(t)$  [mM]')
# plt.xlim(*X_LIM_S); plt.ylim(*Y_LIM_M)
# plt.title('AIF via Reference-Tissue Method (stabilized)')
# plt.legend(); plt.tight_layout()
# plt.savefig(SAVE_DIR / "aif_reference_tissue.png", dpi=160)
# plt.show()
#
# print(f"Saved figures in: {SAVE_DIR.resolve()}")
