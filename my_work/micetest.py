import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import re

# Set your data path
root = Path("/home/siq/Data_ACEMRI")

# Find all 3D files and sort them by time/number
dce_files = sorted(list(root.rglob("*.img")) + list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz")))

if not dce_files:
    print("No DCE files found!")
    exit()

print(f"Found {len(dce_files)} files")

# Extract time curves from each file
time_curves = []

for i, file_path in enumerate(dce_files):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()

        if len(data.shape) == 3:  # 3D volume
            # Extract mean intensity from center ROI
            x, y, z = data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2
            roi_intensity = np.mean(data[x - 2:x + 2, y - 2:y + 2, z])
            time_curves.append(roi_intensity)
            print(f"File {i + 1}: {file_path.name} - Intensity: {roi_intensity:.2f}")

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Plot the time curve
if time_curves:
    plt.figure(figsize=(10, 6))
    plt.plot(time_curves, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Time Point / File Index')
    plt.ylabel('Signal Intensity')
    plt.title('DCE-MRI Time-Intensity Curve\n(Multiple 3D Files)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"DCE-MRI Time-Intensity Curve\n(Multiple 3D Files).png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nAnalysis:")
    print(f"Number of time points: {len(time_curves)}")
    print(f"Mean intensity: {np.mean(time_curves):.2f}")
    print(f"Max intensity: {np.max(time_curves):.2f}")
    print(f"Min intensity: {np.min(time_curves):.2f}")
else:
    print("No valid time curves extracted")

###################################################################3

#
# # Set your data path
# root = Path("/home/siq/Data_ACEMRI")
#
# # Find and load first file
# dce_files = list(root.rglob("*.img")) + list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
# if not dce_files:
#     print("No DCE files found!")
#     exit()
#
# img = nib.load(dce_files[0])
# data = img.get_fdata()
#
# print(f"Data shape: {data.shape}")
# print(f"Data range: {data.min():.2f} to {data.max():.2f}")
#
# # Display three orthogonal slices
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
# # Axial slice (xy-plane)
# slice_z = data.shape[2] // 2
# axes[0].imshow(data[:, :, slice_z], cmap='gray')
# axes[0].set_title(f'Axial Slice (z={slice_z})')
#
# # Sagittal slice (yz-plane)
# slice_x = data.shape[0] // 2
# axes[1].imshow(data[slice_x, :, :].T, cmap='gray', origin='lower')
# axes[1].set_title(f'Sagittal Slice (x={slice_x})')
#
# # Coronal slice (xz-plane)
# slice_y = data.shape[1] // 2
# axes[2].imshow(data[:, slice_y, :].T, cmap='gray', origin='lower')
# axes[2].set_title(f'Coronal Slice (y={slice_y})')
#
# plt.tight_layout()
# plt.show()
#
# # Show intensity histogram
# plt.figure(figsize=(10, 4))
# plt.hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
# plt.xlabel('Intensity')
# plt.ylabel('Frequency')
# plt.title('Intensity Histogram')
# plt.grid(True, alpha=0.3)
# plt.show()
#
# ###################################################################################3
#
#
# # Set your data path
# root = Path("/home/siq/Data_ACEMRI")
#
# # Find all files and check their properties
# dce_files = list(root.rglob("*.img")) + list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
#
# print("File analysis:")
# print("=" * 50)
#
# for i, file_path in enumerate(dce_files):
#     try:
#         img = nib.load(file_path)
#         data = img.get_fdata()
#         print(f"{i + 1:2d}. {file_path.name}")
#         print(f"    Shape: {data.shape}")
#         print(f"    Range: {data.min():.2f} to {data.max():.2f}")
#         print(f"    Mean: {data.mean():.2f}")
#         print()
#
#     except Exception as e:
#         print(f"{i + 1:2d}. {file_path.name} - Error: {e}")
#         print()
#
# # If all files have same shape, they might be time points
# if len(dce_files) > 1:
#     shapes = []
#     for file_path in dce_files:
#         try:
#             img = nib.load(file_path)
#             shapes.append(img.get_fdata().shape)
#         except:
#             pass
#
#     if all(shape == shapes[0] for shape in shapes):
#         print("All files have the same shape - likely a time series!")
#         print("Run Option 1 script to create time-intensity curve.")