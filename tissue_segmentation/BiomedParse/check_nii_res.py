import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

# Read the colon.nii file from the same directory
file_path = os.path.join(os.path.dirname(__file__), "colon.nii")
image = sitk.ReadImage(file_path)

# Convert SimpleITK image to numpy array
array = sitk.GetArrayFromImage(image)

# Check values in the array
unique_values = np.unique(array)
print(f"Unique values: {unique_values}")

# Count the number of 0s and 1s
zeros_count = np.sum(array == 0)
ones_count = np.sum(array == 1)
print(f"Number of 0s: {zeros_count}")
print(f"Number of 1s: {ones_count}")
print(f"Total pixels: {array.size}")
print(f"Percentage of 1s: {ones_count / array.size * 100:.2f}%")

# If it's a 3D volume, display some slices to visualize the distribution
if len(array.shape) == 3:
    z_middle = array.shape[0] // 2
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(array[z_middle, :, :], cmap='gray')
    plt.title(f'Axial Slice (z={z_middle})')
    
    plt.subplot(132)
    plt.imshow(array[:, array.shape[1]//2, :], cmap='gray')
    plt.title(f'Coronal Slice (y={array.shape[1]//2})')
    
    plt.subplot(133)
    plt.imshow(array[:, :, array.shape[2]//2], cmap='gray')
    plt.title(f'Sagittal Slice (x={array.shape[2]//2})')
    
    plt.tight_layout()
    plt.savefig('colon_visualization.png')
    plt.show()

print("Analysis complete")
