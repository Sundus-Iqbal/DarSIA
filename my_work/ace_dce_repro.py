#!/usr/bin/env python3
"""
ACE-MRI vs DCE-MRI — Full script with all fixes
- Analyze (.img/.hdr) loader
- ROI .mat loader (extract 'bw')
- Mask alignment/transposition for (X,Y,T) data
- Non-uniform reference tissue AIF (Tikhonov) for DCE (baseline) and ACE (all frames)
- DCE fit (baseline) and ACE fit (all frames; estimates T10, B1)
- Saves PNG plots (no interactive show)

Requirements:
    pip install numpy nibabel scipy matplotlib
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # save PNGs instead of opening windows
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
import scipy.io as sio

# ==========================================================
# === Configuration ========================================
# ==========================================================
DATA_ROOT = Path("/home/siq/Data_ACEMRI")
MOUSE = "M01_Post"      # e.g. "M01_Pre", "M01_Post"
SLICE = "Slice5fid"     # e.g. "Slice5fid"
OUTDIR = Path("./ace_outputs") / f"{MOUSE}_{SLICE}"
OUTDIR.mkdir(parents=True, exist_ok=True)

R1_RELAXIVITY = 4.3  # Gd-DTPA relaxivity [mM^-1 s^-1]
T10_REF_SEC = 1.8    # nominal pre-contrast T1 for reference tissue (DCE)
KTRANS_REF_MIN = 0.11  # min^-1 (reference tissue)
VE_REF = 0.20
VP_REF = 0.02

# ==========================================================
# === Schedule & Models ====================================
# ==========================================================
def build_ace_schedule():
    """Recreate 9-segment ACE schedule: FA/TR per frame and dt per frame."""
    FA_deg = [10, 20, 5, 10, 30, 2, 10, 80, 10]
    TR_ms  = [12, 12, 12, 12, 12, 12, 12, 100, 12]
    nfrm   = [40,  5,  5,  5,   5,  5,  5,   3,  5]
    FA = np.concatenate([np.deg2rad(np.full(n, fa)) for fa, n in zip(FA_deg, nfrm)])
    TR = np.concatenate([np.full(n, tr/1000.0) for tr, n in zip(TR_ms, nfrm)])  # seconds
    dt_vec = np.concatenate([
        np.full(n, 5.4) if fa != 80 else np.full(n, 45.0)
        for fa, n in zip(FA_deg, nfrm)
    ])
    TE = 3.83e-3
    return FA, TR, TE, dt_vec

def time_from_dt(dt_vec):
    t = np.cumsum(np.asarray(dt_vec, float))
    t -= t[0]
    return t

def spgr_signal(R1, S0, FA, TR, TE=0.0, T2star=np.inf, B1=1.0):
    E1 = np.exp(-R1 * TR)
    FAeff = B1 * FA
    num = (1.0 - E1) * np.sin(FAeff)
    den = (1.0 - E1 * np.cos(FAeff))
    s = S0 * (num / np.maximum(den, 1e-12))
    if np.isfinite(T2star) and T2star > 0:
        s = s * np.exp(-TE / T2star)
    return s

def invert_spgr_for_R1(S, S0, FA, TR):
    """Approximate inversion of SPGR for R1 (1/T1)."""
    y = np.clip(S / np.maximum(S0, 1e-12), 0, None)
    sinA, cosA = np.sin(FA), np.cos(FA)
    denom = (y * cosA - sinA)
    E1 = np.where(np.abs(denom) > 1e-12, (y - sinA) / np.maximum(denom, 1e-12), 0.0)
    E1 = np.clip(E1, 1e-6, 0.999999)
    R1 = -np.log(E1) / np.maximum(TR, 1e-9)
    return R1

def concentration_from_R1(R1_t, R10, r1=R1_RELAXIVITY):
    return (R1_t - R10) / r1

def tofts_concentration(Cp, dt, Ktrans, ve, vp=0.0):
    """Discrete Tofts model with uniform dt (passed in as scalar)."""
    n = len(Cp)
    k = Ktrans / max(ve, 1e-12)
    t = np.arange(n) * dt
    kernel = np.exp(-k * t) * dt
    conv = fftconvolve(Cp, kernel, mode='full')[:n]
    Ct = vp * Cp + Ktrans * conv
    return Ct




# ==========================================================
# === Non-uniform Reference-Tissue AIF =====================
# ==========================================================
def build_nonuniform_conv_matrix(t, Ktrans_ref, ve_ref, vp_ref, lam=1e-2, smooth_order=1):
    """
    Ct_ref(t_i) = vp_ref * Cp(t_i) + Ktrans_ref * sum_{j<=i} Cp(t_j) * exp(-Ktrans_ref/ve_ref*(t_i-t_j)) * Δt_j
    Non-uniform sampling matrix (lower triangular). Returns A, dt, L (smoothing operator).
    """
    t = np.asarray(t, float)
    n = t.size
    dt = np.empty(n)
    dt[0] = t[0] if n > 0 else 0.0
    if n > 1:
        dt[1:] = np.diff(t)

    k = Ktrans_ref / max(ve_ref, 1e-12)
    A = np.zeros((n, n), float)
    for i in range(n):
        tau = t[i] - t[:i+1]
        A[i, :i+1] = Ktrans_ref * np.exp(-k * tau) * dt[:i+1]
        A[i, i] += vp_ref

    if smooth_order == 1:
        L = np.eye(n) - np.vstack([np.zeros((1, n)), np.eye(n-1, n)])
        L = L[1:]  # first difference
    else:
        D1 = np.eye(n) - np.vstack([np.zeros((1, n)), np.eye(n-1, n)])
        D1 = D1[1:]
        L = D1[1:] - D1[:-1]       # second difference
    return A, dt, L

def estimate_aif_from_reference_nonuniform(t, Ct_ref, Ktrans_ref_min=0.11, ve_ref=0.20, vp_ref=0.02,
                                           lam=1e-2, smooth_order=1):
    """Solve for Cp(t) at non-uniform times t from Ct_ref(t) (min^-1 internally converted to s^-1)."""
    Ktrans_ref = Ktrans_ref_min / 60.0  # s^-1
    A, dt, L = build_nonuniform_conv_matrix(t, Ktrans_ref, ve_ref, vp_ref, lam=lam, smooth_order=smooth_order)
    ATA = A.T @ A + lam * (L.T @ L)
    ATy = A.T @ Ct_ref
    Cp = np.linalg.solve(ATA, ATy)
    return np.clip(Cp, 0, None)

# ==========================================================
# === I/O: Dynamic & ROIs ==================================
# ==========================================================
def load_dynamic_images():
    """Load dynamic 3D Analyze image (X,Y,T) from .img/.hdr."""
    dce_img_path = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}.img"
    hdr_path = dce_img_path.with_suffix(".hdr")
    if not dce_img_path.exists():
        raise FileNotFoundError(f"Dynamic image not found: {dce_img_path}")
    if not hdr_path.exists():
        raise FileNotFoundError(f"Header not found: {hdr_path}")
    img = nib.load(str(dce_img_path))
    data = img.get_fdata().astype(np.float32)
    # Some Analyze stacks may be (X, Y, Z, T) with Z=1; squeeze that:
    if data.ndim == 4 and data.shape[2] == 1:
        data = data[:, :, 0, :]
    if data.ndim != 3:
        raise ValueError(f"Expected (X,Y,T) or (X,Y,1,T); got shape {data.shape}")
    print(f"Loaded dynamic image {data.shape} from {dce_img_path}")
    return data

def load_roi_mat():
    """Load tumor and muscle masks from MATLAB .mat (extract 'bw' from ROI struct)."""
    tumor_mat_path = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_tumor.mat"
    muscle_mat_path = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_muscle.mat"
    if not tumor_mat_path.exists():
        raise FileNotFoundError(tumor_mat_path)
    if not muscle_mat_path.exists():
        raise FileNotFoundError(muscle_mat_path)

    tmat = sio.loadmat(tumor_mat_path)
    mmat = sio.loadmat(muscle_mat_path)

    tkey = next(k for k in tmat.keys() if not k.startswith("__"))
    mkey = next(k for k in mmat.keys() if not k.startswith("__"))

    tstruct = tmat[tkey]
    mstruct = mmat[mkey]

    if not hasattr(tstruct, "dtype") or "bw" not in tstruct.dtype.names:
        raise ValueError("Tumor .mat missing 'bw'")
    if not hasattr(mstruct, "dtype") or "bw" not in mstruct.dtype.names:
        raise ValueError("Muscle .mat missing 'bw'")

    tumor_bw = np.array(tstruct["bw"][0, 0], dtype=bool)
    muscle_bw = np.array(mstruct["bw"][0, 0], dtype=bool)

    print(f"Loaded ROIs: tumor {tumor_bw.shape}, muscle {muscle_bw.shape}")
    return tumor_bw, muscle_bw

def prepare_roi_masks(dyn_3d, mask_tumor_2d, mask_muscle_2d):
    """
    Align 2D masks (possibly transposed) to (X,Y,T) dynamic data.
    We embed masks as 3D (X,Y,T) by repeating along time, then reduce to 2D again.
    """
    X, Y, T = dyn_3d.shape
    # Transpose if needed (your masks often arrive as (66,100) vs data (100,66))
    def align2d(msk):
        msk = np.array(msk, dtype=bool)
        if msk.shape == (Y, X):  # transposed
            print("Transposing ROI to match MRI orientation...")
            msk = msk.T
        if msk.shape != (X, Y):
            # center crop/pad
            out = np.zeros((X, Y), bool)
            sx = max((X - msk.shape[0]) // 2, 0)
            sy = max((Y - msk.shape[1]) // 2, 0)
            ex = min(sx + msk.shape[0], X)
            ey = min(sy + msk.shape[1], Y)
            out[sx:ex, sy:ey] = msk[:ex - sx, :ey - sy]
            msk = out
        return msk

    tumor2d = align2d(mask_tumor_2d)
    musc2d  = align2d(mask_muscle_2d)

    # Sanity
    if tumor2d.sum() == 0:
        raise ValueError("Aligned tumor ROI is empty.")
    if musc2d.sum() == 0:
        raise ValueError("Aligned muscle ROI is empty.")

    print(f"Aligned ROIs -> tumor {tumor2d.shape} sum={tumor2d.sum()}, muscle {musc2d.shape} sum={musc2d.sum()}")
    return tumor2d, musc2d

# ==========================================================
# === Fitting functions ====================================
# ==========================================================
def fit_voxel_DCE(S_t, FA_b, TR_b, Cp_b, dt_b, T10=1.8, r1=R1_RELAXIVITY):
    """Fit DCE params on baseline frames for a single voxel time-series."""
    # Estimate S0 from first few baseline points
    S0 = float(np.mean(S_t[:5]))
    R1 = invert_spgr_for_R1(S_t, S0, FA_b, TR_b)
    R10 = 1.0 / max(T10, 1e-6)
    Ct = concentration_from_R1(R1, R10, r1=r1)

    def resid(theta):
        Kt, ve, vp = theta
        Ct_pred = tofts_concentration(Cp_b, dt_b, Kt, ve, vp)
        return Ct_pred - Ct

    x0 = np.array([0.2/60.0, 0.3, 0.05])
    bounds = ([0.001/60.0, 0.001, 0.001], [2.0/60.0, 1.0, 0.5])
    res = least_squares(resid, x0, bounds=bounds, max_nfev=5000)
    return res.x  # Ktrans(s^-1), ve, vp

def fit_voxel_ACE(S_t, FA, TR, Cp_full, dt_uniform=5.4, r1=R1_RELAXIVITY):
    """Fit ACE params on all frames: (Ktrans, ve, vp, T10, B1)."""
    S0 = float(np.mean(S_t[:5]))
    def resid(theta):
        Kt, ve, vp, T10, B1 = theta
        R10 = 1.0 / max(T10, 1e-6)
        Ct = tofts_concentration(Cp_full, dt_uniform, Kt, ve, vp)
        R1 = R10 + r1 * Ct
        Sp = spgr_signal(R1, S0, FA, TR, TE=3.83e-3, T2star=np.inf, B1=B1)
        return Sp - S_t
    x0 = np.array([0.2/60.0, 0.3, 0.05, 1.8, 1.0])
    bounds = ([0.001/60.0, 0.001, 0.001, 0.5, 0.5], [2.0/60.0, 1.0, 0.5, 4.0, 1.5])
    res = least_squares(resid, x0, bounds=bounds, max_nfev=8000)
    return res.x  # (Ktrans, ve, vp, T10, B1)

# ==========================================================
# === Main ==================================================
# ==========================================================
def main():
    # Load data and ROIs
    dyn = load_dynamic_images()                # (X,Y,T) e.g. (100,66,78)
    tumor_bw, muscle_bw = load_roi_mat()       # 2D masks (likely 66x100 or 100x66)
    tumor2d, muscle2d = prepare_roi_masks(dyn, tumor_bw, muscle_bw)  # (X,Y) both

    # Build schedule
    FA, TR, TE, dt_vec = build_ace_schedule()
    T = dyn.shape[2]
    if T != FA.size:
        # If mismatch, truncate to min length
        n = min(T, FA.size)
        print(f"Warning: time frames {T} != schedule {FA.size}. Using first {n} frames.")
        dyn = dyn[:, :, :n]
        FA, TR, dt_vec = FA[:n], TR[:n], dt_vec[:n]
        T = n

    # Indices for DCE baseline frames
    baseline_idx = np.where((np.isclose(FA, np.deg2rad(10.0))) & (np.isclose(TR, 0.012)))[0]
    if baseline_idx.size < 10:
        print("Warning: few baseline frames detected.")

    # ==== AIF (DCE): non-uniform times only at baseline ====
    t_b = time_from_dt(np.full(baseline_idx.size, 5.4))  # uniform 5.4s sample for baseline subset
    # Reference ROI time curve on baseline frames
    x_ref, y_ref = np.where(muscle2d)
    S_ref_b = dyn[x_ref, y_ref, :][:, baseline_idx].mean(axis=0)  # (Tb,)
    S0_ref_b = float(S_ref_b[:5].mean())
    R1_ref_b = invert_spgr_for_R1(S_ref_b, S0_ref_b, FA[baseline_idx], TR[baseline_idx])
    R10_ref  = 1.0 / T10_REF_SEC
    Ct_ref_b = concentration_from_R1(R1_ref_b, R10_ref, r1=R1_RELAXIVITY)
    Cp_b = estimate_aif_from_reference_nonuniform(t_b, Ct_ref_b,
             Ktrans_ref_min=KTRANS_REF_MIN, ve_ref=VE_REF, vp_ref=VP_REF, lam=1e-2, smooth_order=1)

    # ==== AIF (ACE): non-uniform times across all frames ====
    t_full = time_from_dt(dt_vec)
    S_ref_full = dyn[x_ref, y_ref, :].mean(axis=0)  # (T,)
    S0_ref_full = float(S_ref_full[:5].mean())
    R1_ref_full = invert_spgr_for_R1(S_ref_full, S0_ref_full, FA, TR)
    Ct_ref_full = concentration_from_R1(R1_ref_full, R10_ref, r1=R1_RELAXIVITY)
    Cp_full = estimate_aif_from_reference_nonuniform(t_full, Ct_ref_full,
             Ktrans_ref_min=KTRANS_REF_MIN, ve_ref=VE_REF, vp_ref=VP_REF, lam=1e-2, smooth_order=1)

    # ==== Pick a tumor voxel (center of ROI) and fit ====
    x_t, y_t = np.where(tumor2d)
    if x_t.size == 0:
        raise ValueError("Tumor mask empty after alignment.")
    i0, j0 = x_t[len(x_t)//2], y_t[len(y_t)//2]   # middle ROI voxel
    S_t = dyn[i0, j0, :]  # (T,)

    # DCE fit (baseline frames only)
    S_base = S_t[baseline_idx]
    FA_b, TR_b = FA[baseline_idx], TR[baseline_idx]
    Kt_dce, ve_dce, vp_dce = fit_voxel_DCE(S_base, FA_b, TR_b, Cp_b, dt_b=5.4, T10=T10_REF_SEC, r1=R1_RELAXIVITY)

    # ACE fit (all frames)
    Kt_ace, ve_ace, vp_ace, T10_ace, B1_ace = fit_voxel_ACE(S_t, FA, TR, Cp_full, dt_uniform=5.4, r1=R1_RELAXIVITY)


    total_time = 600  # seconds
    T = len(S_t)
    time_axis = np.linspace(0, total_time, T)  # evenly spread across 600 s
    baseline_tumor = np.mean(S_t[:5])  # baseline (pre-contrast)
    normalized_curve_tumor = S_t / baseline_tumor  # normalize to baseline ≈ 1

    # DCE prediction
    S0_v = float(np.mean(S_t[:5]))
    Ct_pred_b = tofts_concentration(Cp_b, 5.4, Kt_dce, ve_dce, vp_dce)
    R1_pred_b = (1.0 / T10_REF_SEC) + R1_RELAXIVITY * Ct_pred_b
    S_pred_dce = spgr_signal(R1_pred_b, S0_v, FA_b, TR_b, TE=3.83e-3, T2star=np.inf, B1=1.0)
    S_pred_dce_norm = S_pred_dce / np.mean(S_pred_dce[:5])

    # ACE prediction
    Ct_pred_full = tofts_concentration(Cp_full, 5.4, Kt_ace, ve_ace, vp_ace)
    R1_pred_full = (1.0 / max(T10_ace, 1e-6)) + R1_RELAXIVITY * Ct_pred_full
    S_pred_ace = spgr_signal(R1_pred_full, S0_v, FA, TR, TE=3.83e-3, T2star=np.inf, B1=B1_ace)
    S_pred_ace_norm = S_pred_ace / np.mean(S_pred_ace[:5])

    # Create DCE time axis that matches baseline frames in 0–600 s scale
    t_b = np.linspace(0, total_time, len(S_pred_dce_norm))

    from scipy.signal import fftconvolve

    # ==== Plot normalized tumor enhancement ====
    plt.figure(figsize=(7, 4))
    plt.plot(time_axis, normalized_curve_tumor, "k", label="Measured (ACE dynamic)")
    plt.plot(t_b, S_pred_dce_norm, "b--", label="DCE fit (baseline)")
    plt.plot(time_axis, S_pred_ace_norm, "g--", label="ACE fit (all frames)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Signal Intensity")
    plt.title(f"Normalized Voxel Enhancement @ ({i0}, {j0})")
    plt.legend()
    plt.xlim(0, 600)
    plt.ylim(0.5, 2.0)
    plt.tight_layout()
    plt.savefig(OUTDIR / "voxel_enhancement_normalized.png", dpi=300)
    plt.close()

    # ==== Plot normalized AIFs ====
    t_full = np.linspace(0, total_time, len(Cp_full))
    t_b = np.linspace(0, total_time, len(Cp_b))
    plt.figure(figsize=(7, 4))
    plt.plot(t_b, Cp_b / np.max(Cp_b), "b", label="DCE AIF (normalized)")
    plt.plot(t_full, Cp_full / np.max(Cp_full), "g", label="ACE AIF (normalized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Cp")
    plt.title("AIF from Reference Tissue (0–600 s)")
    plt.legend()
    plt.xlim(0, 600)
    plt.tight_layout()
    plt.savefig(OUTDIR / "aif_reference_normalized.png", dpi=300)
    plt.close()

    # ==== Print parameters ====
    print("\n=== Fitted parameters (single voxel) ===")
    print(f"DCE: Ktrans={Kt_dce*60:.3f} min^-1, ve={ve_dce:.3f}, vp={vp_dce:.3f}")
    print(f"ACE: Ktrans={Kt_ace*60:.3f} min^-1, ve={ve_ace:.3f}, vp={vp_ace:.3f}, T10={T10_ace:.2f} s, B1={B1_ace:.2f}")
    print(f"Outputs saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
