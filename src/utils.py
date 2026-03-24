"""
Utility functions for image processing and evaluation metrics
"""
import numpy as np
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import io
import lpips


def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images
    Args:
        img1, img2: numpy arrays with values in [0, 1] or [0, 255]
    """
    # Ensure images are in [0, 255] range
    if img1.max() <= 1.0:
        img1 = img1 * 255
    if img2.max() <= 1.0:
        img2 = img2 * 255
    
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255 ** 2 / mse)


def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images
    Args:
        img1, img2: numpy arrays with values in [0, 1] or [0, 255]
    """
    # Normalize to [0, 1]
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Convert to grayscale if RGB
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = np.dot(img1[..., :3], [0.299, 0.587, 0.114])
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = np.dot(img2[..., :3], [0.299, 0.587, 0.114])
    
    return structural_similarity(img1, img2, data_range=1.0)


def calculate_lpips(img1, img2, device='cpu'):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) between two images
    Args:
        img1, img2: numpy arrays with values in [0, 1] or [0, 255]
        device: 'cpu' or 'cuda'
    """
    # Initialize model
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)
    
    # Convert to torch tensors
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Ensure shape is (H, W, C)
    if len(img1.shape) == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    if len(img2.shape) == 2:
        img2 = np.stack([img2] * 3, axis=-1)
    
    # Convert to (1, C, H, W) and torch tensor
    img1_t = torch.from_numpy(img1.transpose(2, 0, 1)[None, ...]).float().to(device)
    img2_t = torch.from_numpy(img2.transpose(2, 0, 1)[None, ...]).float().to(device)
    
    # Scale to [-1, 1]
    img1_t = img1_t * 2 - 1
    img2_t = img2_t * 2 - 1
    
    with torch.no_grad():
        lpips_val = loss_fn_alex(img1_t, img2_t).item()
    
    return lpips_val


def add_gaussian_noise(image, sigma=0.1):
    """
    Add Gaussian noise to image
    Args:
        image: numpy array with values in [0, 1]
        sigma: standard deviation of noise
    Returns:
        noisy image
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image


def apply_gaussian_blur(image, kernel_size=3):
    """
    Apply Gaussian blur
    Args:
        image: numpy array with values in [0, 1] or [0, 255]
        kernel_size: size of blur kernel (3 or 5)
    Returns:
        blurred image
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    if image.max() > 1.0:
        return blurred.astype(np.float32) / 255.0
    return blurred.astype(np.float32)


def denoise_bilateral(image):
    """
    Denoise using bilateral filter
    Args:
        image: numpy array with values in [0, 1] or [0, 255]
    Returns:
        denoised image
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    if image.max() > 1.0:
        return denoised.astype(np.float32) / 255.0
    return denoised.astype(np.float32)


def richardson_lucy_deblur(image, psf=None, num_iterations=10):
    """
    Richardson-Lucy deblurring
    Args:
        image: numpy array
        psf: point spread function
        num_iterations: number of iterations
    Returns:
        deblurred image
    """
    from skimage.restoration import richardson_lucy
    
    if image.max() <= 1.0:
        img_normalized = image
    else:
        img_normalized = image / 255.0
    
    if psf is None:
        # Default PSF (Gaussian blur)
        psf = np.ones((5, 5)) / 25.0
    
    deblurred = richardson_lucy(img_normalized, psf, num_iter=num_iterations)
    return np.clip(deblurred, 0, 1)


def resize_bicubic(image, scale_factor=4):
    """
    Resize image using bicubic interpolation
    Args:
        image: numpy array
        scale_factor: upscaling factor
    Returns:
        resized image
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    height, width = image.shape[:2]
    new_size = (width * scale_factor, height * scale_factor)
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    
    return resized.astype(np.float32) / 255.0

