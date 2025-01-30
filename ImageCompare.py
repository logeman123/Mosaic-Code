from PIL import Image
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load images using Pillow
image1 = Image.open('images/President_Barack_Obama1.jpg').convert('L')
image2 = Image.open('images/President_Barack_Obama2.jpg').convert('L')

# Convert images to numpy arrays
image1_np = np.array(image1)
image2_np = np.array(image2)

# Check if the images have the same size
if image1_np.shape == image2_np.shape:
    # Method 2: SSIM comparison
    similarity_index, _ = ssim(image1_np, image2_np, full=True)
    print(f'Structural Similarity Index: {similarity_index}')
else:
    print("Images are of different sizes and cannot be compared.")