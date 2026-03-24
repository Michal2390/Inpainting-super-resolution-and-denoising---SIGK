"""
Dataset preparation and loading
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm


class SuperResolutionDataset(Dataset):
    """
    Dataset for Super-Resolution task
    Takes high-res images and downsamples them
    """
    def __init__(self, image_dir, scale_factor=4, patch_size=256, augment=False):
        """
        Args:
            image_dir: directory containing high-res images
            scale_factor: upscaling factor (2, 3, or 4)
            patch_size: size of patches to extract
            augment: whether to apply data augmentation
        """
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment
        
        # Get list of image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(Path(image_dir).glob(ext))
        self.image_paths = sorted(self.image_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            # Try PIL as fallback
            img = np.array(Image.open(img_path))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Extract patch
        h, w = img.shape[:2]
        if h >= self.patch_size and w >= self.patch_size:
            # Random crop
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            img = img[top:top + self.patch_size, left:left + self.patch_size]
        else:
            # Resize to patch size
            img = cv2.resize(img, (self.patch_size, self.patch_size))
        
        # Data augmentation
        if self.augment:
            # Random flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            img = np.rot90(img, k)
        
        # Create low-res version
        lr_size = self.patch_size // self.scale_factor
        lr = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        # Upsample back to original size using bicubic for baseline comparison
        lr_upsampled = cv2.resize(lr, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        
        # Convert to tensors (C, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1).copy())
        lr_tensor = torch.from_numpy(lr.transpose(2, 0, 1).copy())
        
        return {
            'lr': lr_tensor,
            'hr': img_tensor,
            'lr_upsampled': torch.from_numpy(lr_upsampled.transpose(2, 0, 1))
        }


class DenoiseDataset(Dataset):
    """
    Dataset for Denoising task
    Takes images and adds Gaussian noise
    """
    def __init__(self, image_dir, sigma=0.1, patch_size=256, augment=False):
        """
        Args:
            image_dir: directory containing clean images
            sigma: standard deviation of Gaussian noise
            patch_size: size of patches to extract
            augment: whether to apply data augmentation
        """
        self.image_dir = image_dir
        self.sigma = sigma
        self.patch_size = patch_size
        self.augment = augment
        
        # Get list of image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(Path(image_dir).glob(ext))
        self.image_paths = sorted(self.image_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.array(Image.open(img_path))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Extract patch
        h, w = img.shape[:2]
        if h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            img = img[top:top + self.patch_size, left:left + self.patch_size]
        else:
            img = cv2.resize(img, (self.patch_size, self.patch_size))
        
        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            k = np.random.randint(0, 4)
            img = np.rot90(img, k)
        
        # Add noise
        noise = np.random.normal(0, self.sigma, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)
        
        # Convert to tensors (C, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1).copy())
        noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1).copy())
        
        return {
            'clean': img_tensor,
            'noisy': noisy_tensor
        }


class DIV2KDataset(Dataset):
    """
    Wrapper for DIV2K dataset
    """
    def __init__(self, image_dir, task='sr', scale_factor=4, sigma=0.1, 
                 patch_size=256, augment=False):
        """
        Args:
            image_dir: directory containing DIV2K images
            task: 'sr' for super-resolution or 'denoise' for denoising
            scale_factor: for SR task
            sigma: for denoise task
            patch_size: size of patches
            augment: data augmentation
        """
        self.task = task
        
        if task == 'sr':
            self.dataset = SuperResolutionDataset(
                image_dir, scale_factor=scale_factor, 
                patch_size=patch_size, augment=augment
            )
        elif task == 'denoise':
            self.dataset = DenoiseDataset(
                image_dir, sigma=sigma,
                patch_size=patch_size, augment=augment
            )
        elif task == 'inpainting':
            self.dataset = InpaintingDataset(
                image_dir, patch_size=patch_size, augment=augment
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def create_dataloaders(train_dir, val_dir, test_dir=None, task='sr', 
                       scale_factor=4, sigma=0.1, batch_size=32, 
                       num_workers=4):
    """
    Create train/val/test dataloaders
    """
    train_dataset = DIV2KDataset(
        train_dir, task=task, scale_factor=scale_factor,
        sigma=sigma, patch_size=256, augment=True
    )
    
    val_dataset = DIV2KDataset(
        val_dir, task=task, scale_factor=scale_factor,
        sigma=sigma, patch_size=256, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    loaders = {'train': train_loader, 'val': val_loader}
    
    if test_dir is not None:
        test_dataset = DIV2KDataset(
            test_dir, task=task, scale_factor=scale_factor,
            sigma=sigma, patch_size=256, augment=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True
        )
        loaders['test'] = test_loader
    
    return loaders


def download_div2k(save_dir='data/', split='train', downsample=False):
    """
    Download DIV2K dataset
    Args:
        save_dir: directory to save the dataset
        split: 'train' or 'val'
        downsample: whether to download downsampled versions
    """
    os.makedirs(save_dir, exist_ok=True)
    
    base_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/'
    
    if split == 'train':
        url = base_url + 'DIV2K_train_HR.zip'
        filename = 'DIV2K_train_HR.zip'
    elif split == 'val':
        url = base_url + 'DIV2K_valid_HR.zip'
        filename = 'DIV2K_valid_HR.zip'
    else:
        raise ValueError(f"Unknown split: {split}")
    
    filepath = os.path.join(save_dir, filename)
    
    if os.path.exists(filepath):
        print(f"{filename} already exists")
        return
    
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    
    print("Done!")


class InpaintingDataset(Dataset):
    def __init__(self, hr_dir, patch_size=256, augment=True):
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))])
        self.patch_size = patch_size
        self.augment = augment

    def __getitem__(self, idx):
        img = np.array(Image.open(self.hr_paths[idx]).convert('RGB')) / 255.0

        # Crop to patch_size
        h, w, _ = img.shape
        top = np.random.randint(0, h - self.patch_size + 1) if h > self.patch_size else 0
        left = np.random.randint(0, w - self.patch_size + 1) if w > self.patch_size else 0
        img = img[top:top + self.patch_size, left:left + self.patch_size]

        if self.augment:
            if np.random.random() > 0.5: img = np.flip(img, axis=1)
            if np.random.random() > 0.5: img = np.rot90(img)

        # Tworzenie maski (1 dla dziury, 0 dla obrazu)
        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

        # Wybieramy losowo wielkość dziury zgodnie z PDF (3x3 lub 32x32)
        hole_size = np.random.choice([3, 32])
        y = np.random.randint(0, self.patch_size - hole_size)
        x = np.random.randint(0, self.patch_size - hole_size)
        mask[y:y + hole_size, x:x + hole_size] = 1.0

        # Tworzenie uszkodzonego obrazu
        masked_img = img.copy()
        masked_img[mask == 1] = 0.0  # Zamalowujemy dziurę na czarno

        return {
            'masked': torch.from_numpy(masked_img.transpose(2, 0, 1).copy()).float(),
            'clean': torch.from_numpy(img.transpose(2, 0, 1).copy()).float(),
            'mask': torch.from_numpy(mask).float()  # Maska przyda się do ewaluacji
        }

    def __len__(self):
        return len(self.hr_paths)
