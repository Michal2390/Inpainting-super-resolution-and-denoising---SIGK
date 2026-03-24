#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test GPU - Sprawdzenie czy RTX 4070Ti działa
"""
import sys
import os

# Ustaw encoding UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("\n" + "="*70)
print("SPRAWDZENIE KONFIGURACJI GPU - RTX 4070Ti")
print("="*70 + "\n")

# Test PyTorch
try:
    import torch
    print(f"[OK] PyTorch zainstalowany: {torch.__version__}")
except ImportError:
    print("[ERROR] PyTorch nie zainstalowany!")
    sys.exit(1)

# Test CUDA
print(f"\n--- CUDA ---")
print(f"CUDA dostepny: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Nazwa: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Memory info
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        
else:
    print("[ERROR] CUDA nie dostepny!")
    sys.exit(1)

# Test computation
print(f"\n--- TEST OBLICZENIOWY ---")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"[OK] Obliczenia na GPU dziala!")
    print(f"     Wynik tensor shape: {z.shape}")
except Exception as e:
    print(f"[ERROR] Problem z GPU: {e}")
    sys.exit(1)

# Test deep learning
print(f"\n--- TEST MODELU NEURONOWEGO ---")
try:
    import torch.nn as nn
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 100)
            self.fc2 = nn.Linear(100, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleNet().cuda()
    x = torch.randn(32, 10).cuda()
    y = model(x)
    print(f"[OK] Model neuronowy dziala na GPU!")
    print(f"     Input shape: {x.shape}")
    print(f"     Output shape: {y.shape}")
    print(f"     Model device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"[ERROR] Problem z modelem: {e}")
    sys.exit(1)

# Config check
print(f"\n--- KONFIGURACJA PROJEKTU ---")
sys.path.insert(0, 'src')
try:
    from config import TRAINING_CONFIG
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Device: {TRAINING_CONFIG['device']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
except Exception as e:
    print(f"[ERROR] Nie mozna zaladowac config: {e}")

print("\n" + "="*70)
print("[SUCCESS] GPU RTX 4070Ti jest GOTOWY DO TRENOWANIA!")
print("="*70)
print("\nNastepnie uruchom: run.bat i wybierz '3' - Trenowanie\n")

