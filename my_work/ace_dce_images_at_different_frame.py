import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

# ----------------------------------------------------------------------
# CONFIGURATION
DATA_ROOT = Path("/home/siq/Data_ACEMRI")
MOUSE = "M01_Post"
SLICE = "Slice5fid"

# Paths
dce_img_path = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}.img"
roi_mat_path_tumor = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_tumor.mat"
roi_mat_path_muscle = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_muscle.mat"

#------------------------------------------------------------------------------------

roi_mat_path_tumor = Path("/home/siq/Data_ACEMRI/M01_Post/DCE/Slice5fid_tumor.mat")
mat_tumor = loadmat(roi_mat_path_tumor)
print(mat_tumor.keys())# Print keys of the top-level dictionary
# Check the structure of the 'roi' variable if present
if 'roi' in mat_tumor:
    print(type(mat_tumor['roi']))
    print("roi structure shape:", mat_tumor['roi'].shape)
    print("Available fields:")
    print(mat_tumor['roi'].dtype)

# ----------------------------------------------------------------------
# LOAD DCE IMAGE
img = nib.load(str(dce_img_path))
data = img.get_fdata()  # shape: (X, Y, T)
voxel_size = img.header.get_zooms()

print(f"DCE shape: {data.shape}")
print(f"Voxel size: {voxel_size}")

mat_tumor = loadmat(roi_mat_path_tumor)
roi_struct_tumor = mat_tumor['roi'][0, 0]  #
mat_muscle = loadmat(roi_mat_path_muscle)
roi_struct_muscle = mat_muscle['roi'][0, 0]

# binary mask
mask = None

mask_tumor = np.array(roi_struct_tumor['bw'], dtype=bool).T
mask_muscle = np.array(roi_struct_muscle['bw'], dtype=bool).T

print(f"Tumor ROI shape: {mask_tumor.shape}, Muscle ROI shape: {mask_muscle.shape}")

# ----------------------------------------------------------------------
# pick a frame
t_mid = data.shape[2] // 2  #// 2 picks the middle frame in time
frame = data[:, :, t_mid]

# normalize for display
frame_norm = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))

# flip image both vertically and horizontally (rotate 180°)
frame_disp = np.flipud(np.fliplr(frame_norm))  # or: np.rot90(frame_norm, 2)

mask_tumor_disp  = mask_tumor
mask_muscle_disp = mask_muscle

# plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(frame_disp.T, cmap='gray', origin='lower')
ax.contour(mask_tumor_disp.T,  levels=[0.5], colors='r', linewidths=2)
ax.contour(mask_muscle_disp.T, levels=[0.5], colors='b', linewidths=2)
ax.set_title(f"{MOUSE} - {SLICE} (Frame {t_mid})")
ax.axis('off')
plt.tight_layout(); plt.show()

data_disp = np.flipud(np.fliplr(data))
mask_tumor_disp  = mask_tumor
mask_muscle_disp = mask_muscle


# Define ACE-MRI protocol segments
FA =  [10, 20, 5, 10, 30, 2, 10, 80, 10]      # flip angles (deg)
TR =  [12, 12, 12, 12, 12, 12, 12, 100, 12]   # repetition times (ms)
NFR = [40, 5, 5, 5, 5, 5, 5, 3, 5]            # frames per segment
dt  = [5.4 if fa < 80 else 45 for fa in FA]   # temporal resolution (s/frame)

# Representative times (end of each segment)
times_min = [1.7, 3.7, 4.2, 4.7, 5.2, 5.6, 6.0, 7.3, 8.6]
time_labels = ["(FA×NR, TR)"] + [f"{t:.1f} min" for t in times_min]

# Compute cumulative frame indices and pick last frame of each block
rep_frames = np.cumsum(NFR) - 1
frame_ids = np.concatenate([[0], rep_frames])

print("Representative frame indices:", rep_frames)
print("Time labels:", time_labels)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
axes = axes.ravel()

for ax, fid, label in zip(axes, frame_ids, time_labels):
    frame = data_disp[:, :, int(fid)]
    frame_norm = (frame - frame.min()) / (frame.max() - frame.min())
    ax.imshow(frame_norm.T, cmap="gray", origin="lower")

    # overlay ROI only for first baseline panel
    if label.startswith("("):
        ax.contour(mask_tumor_disp.T,  [0.5], colors="r", linewidths=1.5)
        ax.contour(mask_muscle_disp.T, [0.5], colors="b", linewidths=1.5)

    ax.set_title(label, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()








