import os

from scipy.stats import alpha

import darsia as da
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images from folder
image_folder = "/home/siq/c1"
#output_folder =

image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
])

images = []

for filename in image_files:
    # Read image using darsia (returns object with .img as NumPy array)
    img_obj = da.imread(os.path.join(image_folder, filename))
    img_array = img_obj.img
#    print(type(img_array))
#     if np.issubdtype(img_array.dtype, np.floating):
#         img_array = (img_array * 65535).astype(np.uint16)
    print(f"{filename}: dtype={img_array.dtype}, range={img_array.min()}-{img_array.max()}")
    # Downsample / resize
    img_resized = cv2.resize(img_array, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
#    print(np.shape(img_array))
#    print(np.shape(img_resized))
    images.append(img_resized)
    #save_path = os.path.join(output_folder, filename)
# print(f"Original image dtype: {img_array.dtype}")
# print(f"Original value range: {img_array.min()} to {img_array.max()}")
# print(f"Resized image dtype: {img_resized.dtype}")
# print(f"Resized value range: {img_resized.min()} to {img_resized.max()}")
# stack into a single 4D array
image_stack = np.stack(images, axis=0)
print(f"Processed {len(images)} images")
#print("Image stack shape:", image_stack.shape, image_stack.dtype)




x, y = 400, 300
time_curve = image_stack[:, y, x, 0] #intensity at x,y for all images
print("Time curve length:", len(time_curve))

#plot the time curve
plt.rcParams.update({
    "axes.titlesize": 24,     # Title font size
    "axes.labelsize": 22,     # X and Y label size
    "xtick.labelsize": 20,    # X tick labels
    "ytick.labelsize": 20,    # Y tick labels
    #"legend.fontsize": 14     # Legend font size
})

# Example plot
plt.figure(figsize=(12, 8))
plt.plot(time_curve, marker="o")
plt.xlabel("Frame index")
plt.ylabel("Intensity")
plt.title("Time curve")
plt.grid(True, alpha=0.3)
plt.savefig(f"tick-time_curve_pixel_{x}_{y}.png", dpi=300, bbox_inches='tight')
plt.show()


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import darsia as da

# ----------------------------
# Load images
# ----------------------------
image_folder = "/home/siq/c1"

image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
])

images = []
for filename in image_files:
    img_obj = da.imread(os.path.join(image_folder, filename))
    img_array = img_obj.img

    # If float -> scale to [0,255] for visualization
    if np.issubdtype(img_array.dtype, np.floating):
        img_array = (255 * (img_array - img_array.min()) / (img_array.ptp()+1e-9)).astype(np.uint8)

    # Downsample
    img_resized = cv2.resize(img_array, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    images.append(img_resized)

image_stack = np.stack(images, axis=0)
print(f"Loaded {len(image_stack)} frames with shape {image_stack.shape}")

# ----------------------------
# Setup matplotlib figure
# ----------------------------
plt.ion()  # make interactive plotting work
fig, ax = plt.subplots(figsize=(8,6))

# If grayscale vs RGB
if image_stack.shape[-1] == 3:
    im = ax.imshow(image_stack[0])  # color
else:
    im = ax.imshow(image_stack[0], cmap="gray")  # grayscale

ax.set_title("CO₂ Injection Experiment (frame 0)")

# ----------------------------
# Animation function
# ----------------------------
def update(frame):
    im.set_data(image_stack[frame])
    ax.set_title(f"CO₂ Injection Experiment (frame {frame})")
    return [im]

ani = FuncAnimation(fig, update, frames=len(image_stack), interval=200, blit=False)

plt.show(block=True)  # block until closed







