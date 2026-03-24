"""
Main script to run the entire pipeline
"""
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset import download_div2k, create_dataloaders
from src.models import create_model
from src.train import Trainer
from src.evaluate import evaluate_models


def main():
    """Main pipeline"""
    
    # Configuration
    config = {
        'device': 'cuda', ##if torch.cuda.is_available() else 'cpu',
        'tasks': ['sr', 'denoise', 'inpainting'],  # Tasks to train
        'sr_scale_factor': 4,
        'denoise_sigma': 0.1,
        'batch_size': 32 if torch.cuda.is_available() else 8,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'data_dir': 'data/',
        'checkpoint_dir': 'models/',
        'results_dir': 'results/'
    }
    
    print(f"Configuration:")
    print(f"  Device: {config['device']}")
    print(f"  Tasks: {config['tasks']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print()
    
    # Step 1: Download DIV2K dataset (if needed)
    print("Step 1: Checking dataset...")
    train_dir = os.path.join(config['data_dir'], 'DIV2K_train_HR')
    val_dir = os.path.join(config['data_dir'], 'DIV2K_valid_HR')
    
    if not os.path.exists(train_dir):
        print(f"Downloading training dataset...")
        download_div2k(config['data_dir'], split='train')
    else:
        print(f"Training dataset already exists")
    
    if not os.path.exists(val_dir):
        print(f"Downloading validation dataset...")
        download_div2k(config['data_dir'], split='val')
    else:
        print(f"Validation dataset already exists")
    
    # Step 2: Train models
    for task in config['tasks']:
        print(f"\n{'='*60}")
        print(f"Training {task.upper()} model")
        print(f"{'='*60}")
        
        # Create dataloaders
        loaders = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            task=task,
            scale_factor=config['sr_scale_factor'] if task == 'sr' else 2,
            sigma=config['denoise_sigma'] if task == 'denoise' else 0.1,
            batch_size=config['batch_size']
        )
        
        # Create model
        model = create_model(
            task=task,
            scale_factor=config['sr_scale_factor'] if task == 'sr' else 2,
            device=config['device']
        )
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Create trainer
        checkpoint_dir = os.path.join(config['checkpoint_dir'], task)
        trainer = Trainer(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            task=task,
            device=config['device'],
            lr=config['learning_rate'],
            checkpoint_dir=checkpoint_dir
        )
        
        # Train
        history = trainer.train(num_epochs=config['num_epochs'])
        
        print(f"\nTraining complete for {task}!")
    
    print("\n" + "="*60)
    print("All training complete!")
    print("="*60)
    
    # Step 3: Evaluate models (optional - can be run separately)
    print("\nTo evaluate models, run evaluate.py or use notebooks/evaluation.ipynb")


if __name__ == '__main__':
    main()

