"""
Training script for image restoration models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models import create_model
from utils import calculate_psnr, calculate_ssim

# TensorBoard - opcjonalnie
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        # Pobieramy wytrenowaną sieć VGG16 (tylko część odpowiedzialną za ekstrakcję cech)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # Używamy pierwszych 16 warstw (do warstwy relu3_3) - to standard w Perceptual Loss
        self.vgg = nn.Sequential(*list(vgg.children())[:16]).to(device)

        # Zamrażamy wagi VGG - nie chcemy jej trenować, służy nam tylko jako "sędzia"
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

        # VGG oczekuje specyficznej normalizacji obrazu (średnia i odchylenie z ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, x, y):
        # Normalizacja wejść dla VGG
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # Przepuszczamy wyjście z naszego modelu i prawdziwy obraz przez VGG
        x_features = self.vgg(x)
        y_features = self.vgg(y)

        # Liczymy różnicę między "percepcją" obu obrazów
        return self.criterion(x_features, y_features)


class Trainer:
    """Trainer class for model training and validation"""
    
    def __init__(self, model, train_loader, val_loader, task='sr', 
                 device='cpu', lr=1e-3, checkpoint_dir='models/'):
        """
        Args:
            model: neural network model
            train_loader: training data loader
            val_loader: validation data loader
            task: 'sr' or 'denoise'
            device: torch device
            lr: learning rate
            checkpoint_dir: directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss function
        self.criterion = nn.L1Loss()
        # --- VGG ---
        self.perceptual_loss = PerceptualLoss(device)
        self.perceptual_weight = 0.1  # Waga dla lossa VGG (zwykle używa się 0.1 lub 0.01)
        # --------------
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Tensorboard
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(os.path.join(checkpoint_dir, 'logs'))
        else:
            self.writer = None
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            # Get inputs and targets
            if self.task == 'sr':
                inputs = batch['lr'].to(self.device).float()
                targets = batch['hr'].to(self.device).float()
            elif self.task == 'denoise':
                inputs = batch['noisy'].to(self.device).float()
                targets = batch['clean'].to(self.device).float()
            elif self.task == 'inpainting':
                inputs = batch['masked'].to(self.device).float()
                targets = batch['clean'].to(self.device).float()
            else:
                raise ValueError(f"Unknown task: {self.task}")

             # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Podstawowy błąd L1 (piksel po pikselu)
            l1_loss = self.criterion(outputs, targets)

            # Błąd percepcyjny VGG (struktura i tekstura)
            p_loss = self.perceptual_loss(outputs, targets)

            # Łączymy oba błędy!
            loss = l1_loss + (self.perceptual_weight * p_loss)

            # Backward pass
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Log to tensorboard
            if self.global_step % 100 == 0 and self.writer:
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        psnr_values = []
        ssim_values = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validating', leave=False)
            for batch in progress_bar:
                # Get inputs and targets
                if self.task == 'sr':
                    inputs = batch['lr'].to(self.device).float()
                    targets = batch['hr'].to(self.device).float()
                elif self.task == 'denoise':
                    inputs = batch['noisy'].to(self.device).float()
                    targets = batch['clean'].to(self.device).float()
                elif self.task == 'inpainting':
                    inputs = batch['masked'].to(self.device).float()
                    targets = batch['clean'].to(self.device).float()

                # --- POPRAWIONE WCIĘCIE PONIŻEJ ---

                # Forward pass
                outputs = self.model(inputs)

                # Podstawowy błąd L1 (piksel po pikselu)
                l1_loss = self.criterion(outputs, targets)

                # Błąd percepcyjny VGG (struktura i tekstura)
                p_loss = self.perceptual_loss(outputs, targets)

                # Łączymy oba błędy
                loss = l1_loss + (self.perceptual_weight * p_loss)

                total_loss += loss.item()

                # ----------------------------------

                # Calculate metrics
                for i in range(outputs.shape[0]):
                    pred = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    target = targets[i].cpu().numpy().transpose(1, 2, 0)

                    # Clip to [0, 1]
                    pred = np.clip(pred, 0, 1)

                    psnr = calculate_psnr(pred, target)
                    ssim = calculate_ssim(pred, target)

                    psnr_values.append(psnr)
                    ssim_values.append(ssim)

        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('Loss/val', avg_loss, self.global_step)
            self.writer.add_scalar('PSNR/val', avg_psnr, self.global_step)
            self.writer.add_scalar('SSIM/val', avg_ssim, self.global_step)

        return avg_loss, avg_psnr, avg_ssim
    
    def train(self, num_epochs=50, save_interval=5):
        """Train model for specified number of epochs"""
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            print(f"Train Loss: {train_loss:.6f}")
            
            # Validate
            val_loss, val_psnr, val_ssim = self.validate()
            history['val_loss'].append(float(val_loss))
            history['val_psnr'].append(float(val_psnr))
            history['val_ssim'].append(float(val_ssim))
            print(f"Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
            
            # LR Scheduler step
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best', is_best=True)
        
        if self.writer:
            self.writer.close()
        
        # Save history
        history_file = os.path.join(self.checkpoint_dir, 'history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        
        return history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        if is_best:
            checkpoint_name = f'best_{self.task}_model.pth'
        else:
            checkpoint_name = f'{self.task}_model_epoch_{epoch}.pth'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded: {checkpoint_path}")


def train_model(task='sr', data_dir='data/DIV2K_train_HR/', 
                scale_factor=4, sigma=0.1, batch_size=32,
                num_epochs=50, device='cpu', checkpoint_dir='models/'):
    """
    Main training function
    """
    print(f"Training {task.upper()} model")
    print(f"Device: {device}")
    
    # Import here to avoid circular imports
    from dataset import create_dataloaders
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found!")
        print("Please download DIV2K dataset first using download_div2k() function")
        return None
    
    # Create dataloaders
    print("Creating dataloaders...")
    loaders = create_dataloaders(
        data_dir, data_dir, test_dir=None,
        task=task, scale_factor=scale_factor, sigma=sigma,
        batch_size=batch_size
    )
    
    # Create model
    print(f"Creating model...")
    model = create_model(task=task, scale_factor=scale_factor, device=device)
    
    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        task=task,
        device=device,
        checkpoint_dir=os.path.join(checkpoint_dir, task)
    )
    
    # Train
    history = trainer.train(num_epochs=num_epochs)
    
    return trainer, history


if __name__ == '__main__':
    # Example usage
    device = 'cuda' ##if torch.cuda.is_available() else 'cpu'
    
    # Train super-resolution model
    trainer_sr, history_sr = train_model(
        task='sr',
        scale_factor=4,
        num_epochs=50,
        device=device,
        checkpoint_dir='models/'
    )
    
    # Train denoising model
    trainer_dn, history_dn = train_model(
        task='denoise',
        sigma=0.1,
        num_epochs=50,
        device=device,
        checkpoint_dir='models/'
    )






