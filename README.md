import imageio

img1 = imageio.imread('image1.tif')
img2 = imageio.imread('image2.tif')

# If they're RGB, convert to grayscale
if img1.ndim == 3:
    img1 = np.dot(img1[...,:3], [0.299, 0.587, 0.114])
if img2.ndim == 3:
    img2 = np.dot(img2[...,:3], [0.299, 0.587, 0.114])




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
