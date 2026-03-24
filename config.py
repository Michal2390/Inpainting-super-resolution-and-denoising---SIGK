"""
Configuration file for the Image Restoration Project
"""

# Dataset configuration
DATASET_CONFIG = {
    'train_dir': 'data/DIV2K_train_HR/',
    'val_dir': 'data/DIV2K_valid_HR/',
    'test_dir': 'data/DIV2K_valid_HR/',
    'patch_size': 256,
    'num_workers': 4,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 64,  # RTX 4070Ti - można więcej
    'learning_rate': 1e-3,
    'num_epochs': 50,
    'save_interval': 5,  # Save checkpoint every 5 epochs
    'device': 'cuda',  # GPU - RTX 4070Ti
}

# Super-Resolution configuration
SR_CONFIG = {
    'scale_factor': 4,
    'in_channels': 3,
    'out_channels': 3,
    'channels': 32,
    'num_residuals': 4,
}

# Denoising configuration
DENOISE_CONFIG = {
    'sigma': 0.1,  # Standard deviation of Gaussian noise
    'in_channels': 3,
    'out_channels': 3,
    'channels': 32,
    'num_residuals': 8,
}

# Evaluation configuration
EVAL_CONFIG = {
    'compute_lpips': False,  # Set to True for LPIPS calculation (slower)
    'metrics': ['psnr', 'ssim'],
}

# Paths
PATHS = {
    'models': 'models/',
    'results': 'results/',
    'checkpoints': 'models/{task}/',
    'logs': 'models/{task}/logs/',
}

# Hardware configuration
HARDWARE = {
    'use_gpu': True,
    'num_gpus': 1,
    'pin_memory': True,
}

# Model checkpoints
CHECKPOINTS = {
    'sr': 'models/sr/best_sr_model.pth',
    'denoise': 'models/denoise/best_denoise_model.pth',
}

# Baseline methods
BASELINE_METHODS = {
    'sr': {
        'name': 'bicubic_interpolation',
        'params': {'interpolation': 'cv2.INTER_CUBIC'},
    },
    'denoise': {
        'name': 'bilateral_filter',
        'params': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
    },
}


