"""
Evaluation script for model predictions
"""
import torch
import numpy as np
import cv2
from pathlib import Path
import json
import os
from tqdm import tqdm

from models import create_model
from utils import calculate_psnr, calculate_ssim, calculate_lpips
from utils import resize_bicubic, denoise_bilateral, richardson_lucy_deblur


def calculate_telea_baseline(masked_batch, mask_batch):
    batch_size = masked_batch.shape[0]
    results = []

    for i in range(batch_size):
        # 1. Pobieramy i-ty obrazek z batcha, rzutujemy na CPU i NumPy, a ZARAZ POTEM robimy transpose
        img_np = (masked_batch[i].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

        # 2. Pobieramy maskę (z datasetu wychodzi jako [H, W])
        mask_np = (mask_batch[i].cpu().numpy() * 255.0).astype(np.uint8)

        # Opcjonalne zabezpieczenie: jeśli maska ma 3 wymiary np. [1, H, W], spłaszczamy do [H, W]
        if mask_np.ndim == 3:
            mask_np = mask_np[0]

        # 3. Odpalamy OpenCV Telea
        inpainted = cv2.inpaint(img_np, mask_np, 3, cv2.INPAINT_TELEA)

        # 4. Zamieniamy z powrotem na format PyTorch [C, H, W]
        inpainted_float = inpainted.astype(np.float32) / 255.0
        results.append(torch.from_numpy(inpainted_float.transpose(2, 0, 1)))

    # Zwracamy batch i upewniamy się, że trafia na to samo urządzenie co wejście (cuda/cpu)
    return torch.stack(results).to(masked_batch.device)

class Evaluator:
    """Evaluator for model predictions"""
    
    def __init__(self, model, task='sr', device='cpu'):
        """
        Args:
            model: trained model
            task: 'sr' or 'denoise'
            device: torch device
        """
        self.model = model
        self.task = task
        self.device = device
        self.model.eval()

    
    def evaluate_on_dataset(self, test_loader, baseline_fn=None):
        """
        Evaluate model on test dataset
        Args:
            test_loader: test data loader
            baseline_fn: function to compute baseline results
        Returns:
            dict with metrics for model and baseline
        """
        model_results = {
            'psnr': [],
            'ssim': [],
            'lpips': []
        }
        
        baseline_results = {
            'psnr': [],
            'ssim': [],
            'lpips': []
        }
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating', leave=False):
                if self.task == 'sr':
                    inputs = batch['lr'].to(self.device).float()
                    targets = batch['hr'].to(self.device).float()
                    lr_upsampled = batch['lr_upsampled']
                elif self.task == 'denoise':
                    inputs = batch['noisy'].to(self.device).float()
                    targets = batch['clean'].to(self.device).float()
                elif self.task == 'inpainting':
                    inputs = batch['masked'].to(self.device).float()
                    targets = batch['clean'].to(self.device).float()
                    mask = batch['mask']
                
                # Model prediction
                outputs = self.model(inputs)
                
                for i in range(outputs.shape[0]):
                    pred = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    target = targets[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Clip predictions to [0, 1]
                    pred = np.clip(pred, 0, 1)
                    
                    # Model metrics
                    psnr = calculate_psnr(pred, target)
                    ssim = calculate_ssim(pred, target)
                    lpips = calculate_lpips(pred, target, device=self.device)  # Slow, optional
                    
                    model_results['psnr'].append(psnr)
                    model_results['ssim'].append(ssim)
                    model_results['lpips'].append(lpips)
                    
                    # Baseline metrics
                    # Baseline metrics
                    if self.task == 'sr':
                        baseline = lr_upsampled[i].numpy().transpose(1, 2, 0)
                    elif self.task == 'denoise':
                        noisy = inputs[i].cpu().numpy().transpose(1, 2, 0)
                        baseline = denoise_bilateral(noisy)
                    elif self.task == 'inpainting':
                        baseline_batch = calculate_telea_baseline(inputs, mask)
                        # Pobieramy i-ty obrazek z batcha i zmieniamy format na [H, W, C]
                        baseline = baseline_batch[i].cpu().numpy().transpose(1, 2, 0)

                    baseline = np.clip(baseline, 0, 1)
                    
                    baseline = np.clip(baseline, 0, 1)
                    
                    baseline_psnr = calculate_psnr(baseline, target)
                    baseline_ssim = calculate_ssim(baseline, target)
                    baseline_lpips = calculate_lpips(baseline, target, device=self.device)
                    
                    baseline_results['psnr'].append(baseline_psnr)
                    baseline_results['ssim'].append(baseline_ssim)
                    baseline_results['lpips'].append(baseline_lpips)
        
        return model_results, baseline_results
    
    def evaluate_single_image(self, input_path, output_path=None, 
                             compute_baseline=True):
        """
        Evaluate on single image
        Args:
            input_path: path to input image
            output_path: path to save output
            compute_baseline: whether to compute baseline
        Returns:
            dict with results
        """
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Cannot load image: {input_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Process through model
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)[None, ...]).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
        
        output_np = output[0].cpu().numpy().transpose(1, 2, 0)
        output_np = np.clip(output_np, 0, 1)
        
        # Save output if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, (output_np * 255).astype(np.uint8))
        
        return output_np


def evaluate_models(test_dir, task='sr', scale_factor=4, sigma=0.1,
                    model_checkpoints=None, device='cpu', output_dir='results/'):
    """
    Evaluate multiple model checkpoints
    """
    from dataset import DIV2KDataset
    from torch.utils.data import DataLoader
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found!")
        return None
    
    # Create test dataset
    test_dataset = DIV2KDataset(
        test_dir, task=task, scale_factor=scale_factor,
        sigma=sigma, patch_size=256, augment=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    if model_checkpoints is None:
        print("No model checkpoints provided")
        return None
    
    for checkpoint_name, checkpoint_path in model_checkpoints.items():
        print(f"\nEvaluating {checkpoint_name}...")
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        # Load model
        model = create_model(task=task, scale_factor=scale_factor, device=device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        evaluator = Evaluator(model=model, task=task, device=device)
        model_results, baseline_results = evaluator.evaluate_on_dataset(test_loader)

        # Compute statistics
        results[checkpoint_name] = {
            'model': {
                'psnr': {
                    'mean': float(np.mean(model_results['psnr'])),
                    'std': float(np.std(model_results['psnr'])),
                    'min': float(np.min(model_results['psnr'])),
                    'max': float(np.max(model_results['psnr'])),
                    'values': [float(x) for x in model_results['psnr']]
                },
                'ssim': {
                    'mean': float(np.mean(model_results['ssim'])),
                    'std': float(np.std(model_results['ssim'])),
                    'min': float(np.min(model_results['ssim'])),
                    'max': float(np.max(model_results['ssim'])),
                    'values': [float(x) for x in model_results['ssim']]
                },
                'lpips': {
                    'mean': float(np.mean(model_results['lpips'])),
                    'std': float(np.std(model_results['lpips'])),
                    'min': float(np.min(model_results['lpips'])),
                    'max': float(np.max(model_results['lpips'])),
                    'values': [float(x) for x in model_results['lpips']]
                }
            },
            'baseline': {
                'psnr': {
                    'mean': float(np.mean(baseline_results['psnr'])),
                    'std': float(np.std(baseline_results['psnr'])),
                    'min': float(np.min(baseline_results['psnr'])),
                    'max': float(np.max(baseline_results['psnr'])),
                    'values': [float(x) for x in baseline_results['psnr']]
                },
                'ssim': {
                    'mean': float(np.mean(baseline_results['ssim'])),
                    'std': float(np.std(baseline_results['ssim'])),
                    'min': float(np.min(baseline_results['ssim'])),
                    'max': float(np.max(baseline_results['ssim'])),
                    'values': [float(x) for x in baseline_results['ssim']]
                },
                'lpips': {
                    'mean': float(np.mean(baseline_results['lpips'])),
                    'std': float(np.std(baseline_results['lpips'])),
                    'min': float(np.min(baseline_results['lpips'])),
                    'max': float(np.max(baseline_results['lpips'])),
                    'values': [float(x) for x in baseline_results['lpips']]
                }
            }
        }

        # Print results
        print(f"\n{checkpoint_name}:")
        print(f"  Model PSNR: {results[checkpoint_name]['model']['psnr']['mean']:.4f} ± {results[checkpoint_name]['model']['psnr']['std']:.4f}")
        print(f"  Model SSIM: {results[checkpoint_name]['model']['ssim']['mean']:.4f} ± {results[checkpoint_name]['model']['ssim']['std']:.4f}")
        print(f"  Model LPIPS: {results[checkpoint_name]['model']['lpips']['mean']:.4f} ± {results[checkpoint_name]['model']['lpips']['std']:.4f}")
        print(f"  Baseline PSNR: {results[checkpoint_name]['baseline']['psnr']['mean']:.4f} ± {results[checkpoint_name]['baseline']['psnr']['std']:.4f}")
        print(f"  Baseline SSIM: {results[checkpoint_name]['baseline']['ssim']['mean']:.4f} ± {results[checkpoint_name]['baseline']['ssim']['std']:.4f}")
        print(f"  Baseline LPIPS: {results[checkpoint_name]['baseline']['lpips']['mean']:.4f} ± {results[checkpoint_name]['baseline']['lpips']['std']:.4f}")
    
    # Save results to JSON
    results_file = os.path.join(output_dir, f'{task}_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_file}")
    
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example evaluation
    model_checkpoints = {
        'SR_epoch_50': 'models/sr/sr_model_epoch_50.pth',
        'Denoise_epoch_50': 'models/denoise/denoise_model_epoch_50.pth'
    }
    
    # Evaluate SR
    evaluate_models(
        test_dir='../data/DIV2K_valid_HR/',
        task='sr',
        scale_factor=4,
        model_checkpoints={'SR': '../models/sr/best_sr_model.pth'},
        device=device
    )
    
    # Evaluate Denoise
    evaluate_models(
        test_dir='../data/DIV2K_valid_HR/',
        task='denoise',
        sigma=0.1,
        model_checkpoints={'Denoise': '../models/denoise/best_denoise_model.pth'},
        device=device
    )

    print("\nEvaluating Inpainting...")
    evaluate_models(
        test_dir='../data/DIV2K_valid_HR/',
        task='inpainting',
        model_checkpoints={'Inpainting': '../models/inpainting/best_inpainting_model.pth'},
        device=device
    )
