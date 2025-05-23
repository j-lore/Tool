from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

# --- Step 1: Load .tif image and convert to grayscale numpy array ---
def load_tif_as_gray(path):
    img = Image.open(path).convert('L')  # 'L' mode = grayscale
    return np.array(img, dtype=np.float32)

# --- Step 2: Define the contrast sensitivity function (CSF) ---
def csf(fx, fy):
    f = np.sqrt(fx**2 + fy**2)
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)

# --- Step 3: Apply the CSF in frequency domain ---
def apply_csf(image, pixels_per_degree):
    height, width = image.shape
    fx = np.fft.fftfreq(width, d=1/pixels_per_degree)
    fy = np.fft.fftfreq(height, d=1/pixels_per_degree)
    FX, FY = np.meshgrid(fx, fy)
    csf_filter = csf(FX, FY)

    image_fft = fftshift(fft2(image))
    filtered_fft = image_fft * csf_filter
    filtered_image = np.real(ifft2(ifftshift(filtered_fft)))
    return filtered_image

# --- Step 4: Compute perceptual difference using SSO model ---
def sso_jnd(image1, image2, pixels_per_degree=60, minkowski_order=4):
    filtered1 = apply_csf(image1, pixels_per_degree)
    filtered2 = apply_csf(image2, pixels_per_degree)
    diff = np.abs(filtered1 - filtered2)
    jnd = np.power(np.mean(np.power(diff, minkowski_order)), 1/minkowski_order)
    return diff, jnd

# --- Step 5: Load two .tif images ---
img1 = load_tif_as_gray('image1.tif')
img2 = load_tif_as_gray('image2.tif')

if img1.shape != img2.shape:
    raise ValueError("The two images must have the same dimensions!")

# --- Step 6: Run the SSO model ---
diff_map, jnd_value = sso_jnd(img1, img2)

# --- Step 7: Visualize the perceptual difference ---
plt.figure(figsize=(8, 6))
plt.imshow(diff_map, cmap='hot')
plt.title(f'Spatial Standard Observer Difference\nJND Value: {jnd_value:.4f}')
plt.colorbar(label='Perceptual Difference')
plt.axis('off')
plt.show()
######################

# Load images
img1 = load_tif_as_gray('image1.tif')
img2 = load_tif_as_gray('image2.tif')




import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

def spatial_csf(fx, fy):
    """
    A simplified 2D contrast sensitivity function (CSF).
    fx, fy: frequency components (in cycles/degree)
    """
    f = np.sqrt(fx**2 + fy**2)
    csf = 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)
    return csf

def apply_sso(image, pixels_per_degree=60):
    """
    Apply the Spatial Standard Observer model on an image.
    image: grayscale image
    pixels_per_degree: display resolution in pixels per visual degree
    """
    image = image.astype(np.float32)
    height, width = image.shape
    fx = np.fft.fftfreq(width, d=1/pixels_per_degree)
    fy = np.fft.fftfreq(height, d=1/pixels_per_degree)
    FX, FY = np.meshgrid(fx, fy)

    # Compute CSF
    csf = spatial_csf(FX, FY)

    # Transform to frequency domain
    image_fft = fft2(image)
    image_fft = fftshift(image_fft)

    # Apply CSF
    filtered_fft = image_fft * csf

    # Inverse FFT to get filtered image
    filtered = np.real(ifft2(np.fft.ifftshift(filtered_fft)))
    return filtered

def sso_difference(img1, img2, pixels_per_degree=60):
    """
    Compute perceptual difference using simplified SSO model.
    """
    filtered1 = apply_sso(img1, pixels_per_degree)
    filtered2 = apply_sso(img2, pixels_per_degree)
    return np.abs(filtered1 - filtered2)

# Example usage:
# Load two grayscale images (same size)
img1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)

diff = sso_difference(img1, img2)

# Visualize the perceptual difference
plt.imshow(diff, cmap='hot')
plt.title('Perceptual Difference (SSO)')
plt.colorbar()
plt.show()

######################
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def csf(fx, fy):
    f = np.sqrt(fx**2 + fy**2)
    csf_value = 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)
    return csf_value

def apply_csf(image, pixels_per_degree):
    image = image.astype(np.float32)
    height, width = image.shape
    fx = np.fft.fftfreq(width, d=1/pixels_per_degree)
    fy = np.fft.fftfreq(height, d=1/pixels_per_degree)
    FX, FY = np.meshgrid(fx, fy)
    csf_filter = csf(FX, FY)
    image_fft = fftshift(fft2(image))
    filtered_fft = image_fft * csf_filter
    filtered_image = np.real(ifft2(ifftshift(filtered_fft)))
    return filtered_image

def sso_jnd(image1, image2, pixels_per_degree=60, minkowski_order=4):
    filtered1 = apply_csf(image1, pixels_per_degree)
    filtered2 = apply_csf(image2, pixels_per_degree)
    diff = np.abs(filtered1 - filtered2)
    jnd = np.power(np.mean(np.power(diff, minkowski_order)), 1/minkowski_order)
    return jnd

# Example usage:
# Load two grayscale images of the same size
img1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)

# Ensure images are the same size
if img1.shape != img2.shape:
    raise ValueError("Input images must have the same dimensions.")

# Compute the JND value
jnd_value = sso_jnd(img1, img2)
print(f"Perceptual difference (JND): {jnd_value}")
