#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick start script - test models on sample images
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ustaw encoding na UTF-8 dla kompatybilności
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import create_model
from src.utils import calculate_psnr, calculate_ssim, resize_bicubic, denoise_bilateral
import cv2


def test_super_resolution():
    """Test SR model on a sample"""
    print("\n" + "="*60)
    print("Testing Super-Resolution Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dummy input (random image)
    lr_image = np.random.rand(3, 64, 64).astype(np.float32)
    
    # Create model
    sr_model = create_model(task='sr', scale_factor=4, device=device)
    
    # Convert to tensor
    lr_tensor = torch.from_numpy(lr_image[None, ...]).to(device)
    
    # Inference
    with torch.no_grad():
        sr_output = sr_model(lr_tensor)
    
    sr_output_np = sr_output[0].cpu().numpy()
    sr_output_np = np.clip(sr_output_np, 0, 1)
    
    print(f"Input shape (LR):  {lr_image.shape}")
    print(f"Output shape (SR): {sr_output_np.shape}")
    print(f"Expected output:   (3, 256, 256)")
    
    if sr_output_np.shape == (3, 256, 256):
        print("[OK] Super-Resolution model works correctly!")
        return True
    else:
        print("[ERROR] Output shape mismatch")
        return False


def test_denoising():
    """Test Denoise model on a sample"""
    print("\n" + "="*60)
    print("Testing Denoising Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dummy input (noisy image)
    clean_image = np.random.rand(3, 256, 256).astype(np.float32)
    noisy_image = np.clip(clean_image + 0.1 * np.random.randn(3, 256, 256), 0, 1).astype(np.float32)
    
    # Create model
    denoise_model = create_model(task='denoise', device=device)
    
    # Convert to tensor
    noisy_tensor = torch.from_numpy(noisy_image[None, ...]).to(device)
    
    # Inference
    with torch.no_grad():
        denoised_output = denoise_model(noisy_tensor)
    
    denoised_np = denoised_output[0].cpu().numpy()
    denoised_np = np.clip(denoised_np, 0, 1)
    
    print(f"Input shape (noisy):    {noisy_image.shape}")
    print(f"Output shape (denoised): {denoised_np.shape}")
    print(f"Expected output:        (3, 256, 256)")
    
    # Calculate metrics
    psnr = calculate_psnr(denoised_np.transpose(1, 2, 0), clean_image.transpose(1, 2, 0))
    noisy_psnr = calculate_psnr(noisy_image.transpose(1, 2, 0), clean_image.transpose(1, 2, 0))
    
    print(f"\nPSNR - Input (noisy): {noisy_psnr:.2f}")
    print(f"PSNR - Output (denoised): {psnr:.2f}")
    
    if denoised_np.shape == (3, 256, 256):
        print("[OK] Denoising model works correctly!")
        print("     (Note: Model is not yet trained, so PSNR may vary)")
        return True
    else:
        print("[ERROR] Output shape mismatch")
        return False


def test_models_count():
    """Count total parameters in models"""
    print("\n" + "="*60)
    print("Model Architecture Summary")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # SR Model
    sr_model = create_model(task='sr', scale_factor=4, device=device)
    sr_params = sum(p.numel() for p in sr_model.parameters())
    sr_trainable = sum(p.numel() for p in sr_model.parameters() if p.requires_grad)
    
    print(f"\nSuper-Resolution Model:")
    print(f"  Total parameters: {sr_params:,}")
    print(f"  Trainable parameters: {sr_trainable:,}")
    
    # Denoise Model
    denoise_model = create_model(task='denoise', device=device)
    denoise_params = sum(p.numel() for p in denoise_model.parameters())
    denoise_trainable = sum(p.numel() for p in denoise_model.parameters() if p.requires_grad)
    
    print(f"\nDenoising Model:")
    print(f"  Total parameters: {denoise_params:,}")
    print(f"  Trainable parameters: {denoise_trainable:,}")
    
    print(f"\nTotal parameters (both models): {sr_params + denoise_params:,}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("IMAGE RESTORATION PROJECT - QUICK START TEST")
    print("="*60)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nUsing device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        
        results = []
        
        # Test SR
        results.append(("SR Model", test_super_resolution()))
        
        # Test Denoise
        results.append(("Denoise Model", test_denoising()))
        
        # Model summary
        test_models_count()
        
        # Print final summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for test_name, result in results:
            status = "[PASS]" if result else "[FAIL]"
            print(f"{test_name}: {status}")
        
        all_passed = all(r for _, r in results)
        
        print("\n" + "="*60)
        if all_passed:
            print("[SUCCESS] ALL TESTS PASSED - Ready for training!")
            print("\nNext steps:")
            print("1. Download DIV2K dataset: python main.py")
            print("2. Train models: python main.py")
            print("3. Evaluate: jupyter notebook notebooks/evaluation.ipynb")
        else:
            print("[WARNING] SOME TESTS FAILED - Please check the output above")
        print("="*60 + "\n")
        
        return all_passed
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)






