from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from skimage.draw import polygon
import matplotlib.pyplot as plt


from scipy.io import loadmat


DATA_ROOT = Path("/home/siq/Data_ACEMRI")
MOUSE = "M01_Post"
SLICE = "Slice5fid"

dce_img_path = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}.img"
roi_mat_path_tumor = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_tumor.mat"
roi_mat_path_muscle = DATA_ROOT / MOUSE / "DCE" / f"{SLICE}_muscle.mat"

if not dce_img_path.exists():
    raise FileNotFoundError(f"DCE file not found: {dce_img_path}")

img = nib.load(str(dce_img_path))
data = img.get_fdata()  # get image data as a NumPy array ,shape: (X, Y, T)
# print(f"DCE shape: {data.shape}")
# print(f"Voxel size: {img.header.get_zooms()}")


mat_tumor = loadmat(roi_mat_path_tumor)
roi_struct_tumor = mat_tumor['roi'][0, 0]  #
mat_muscle = loadmat(roi_mat_path_muscle)
roi_struct_muscle = mat_muscle['roi'][0, 0]

# binary mask
mask = None

mask_tumor = np.array(roi_struct_tumor['bw'], dtype=bool).T
mask_muscle = np.array(roi_struct_muscle['bw'], dtype=bool).T


#  visualize mask

# plt.imshow(mask_tumor, cmap='gray')
# plt.title("Tumor ROI mask")
# plt.show()
# plt.imshow(mask_muscle, cmap='gray')
# plt.title("Tumor ROI mask")
# plt.show()

#time_curves = []
time_curves_tumor = [data[:, :, t][mask_tumor].mean() for t in range(data.shape[2])] # data[:, :, t] extracts the 2D image at timepoint
time_curves_muscle = [data[:, :, t][mask_muscle].mean() for t in range(data.shape[2])]


total_time = 600  # seconds
T = data.shape[2]
time_axis = np.linspace(0, total_time, T)
baseline_tumor = np.mean(time_curves_tumor[:5])
normalized_curve_tumor = time_curves_tumor / baseline_tumor


baseline_muscle = np.mean(time_curves_muscle[:5])
normalized_muscle = time_curves_muscle / baseline_muscle


plt.rcParams.update({
    "axes.titlesize": 24,     # Title font size
    "axes.labelsize": 22,     # X and Y label size
    "xtick.labelsize": 20,    # X tick labels
    "ytick.labelsize": 20,    # Y tick labels
    "legend.fontsize": 20,     # Legend font size
})
plt.figure(figsize=(12, 8))
plt.plot(time_axis, normalized_curve_tumor , 'r-o', linewidth=2, markersize=6,  label="Tumor ROI")
plt.plot(time_axis, normalized_muscle , 'b-o', linewidth=2, markersize=6,  label="Muscle ROI")
plt.xlabel("Time (s)")
plt.ylabel("Intensity")
plt.legend(loc="best")
#plt.title(f"{MOUSE} {SLICE} Tumor vs Muscle Time-Intensity Curve", fontsize=18)
plt.title(f"Tumor vs Muscle Time-Intensity Curve")
plt.grid(True)
plt.savefig(f"tick-{MOUSE} {SLICE} Tumor vs Muscle Time-Intensity Curve.png", dpi=300, bbox_inches='tight')
plt.show()



