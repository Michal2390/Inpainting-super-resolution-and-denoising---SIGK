#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rozwiązanie dla GPU RTX 4070Ti - jeśli PyTorch CPU
Spróbuj załadować CUDA ręcznie
"""
import sys
import os

# Ustaw encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("\n" + "="*70)
print("GPU SETUP - RTX 4070Ti")
print("="*70 + "\n")

print("Jeśli PyTorch wciąż pokazuje CUDA=False, spróbuj:")
print()
print("1. Zainstaluj NVIDIA CUDA Toolkit 12.1:")
print("   https://developer.nvidia.com/cuda-12-1-0-download-archive")
print()
print("2. Zainstaluj cuDNN:")
print("   https://developer.nvidia.com/cudnn")
print()
print("3. Zainstaluj najnowszy NVIDIA Driver:")
print("   https://www.nvidia.com/Download/driverDetails.aspx")
print()
print("4. Potem uruchom znowu:")
print("   python test_gpu.py")
print()

# Sprawdź zmienne środowiskowe
print("Aktualne zmienne środowiskowe:")
cuda_path = os.getenv('CUDA_PATH')
cudnn_path = os.getenv('CUDNN_PATH')

if cuda_path:
    print(f"CUDA_PATH: {cuda_path}")
else:
    print("CUDA_PATH: Nie ustawiona")

if cudnn_path:
    print(f"CUDNN_PATH: {cudnn_path}")
else:
    print("CUDNN_PATH: Nie ustawiona")

print("\n" + "="*70)
print("Jeśli powyższe ścieżki nie istnieją, zainstaluj:")
print("1. CUDA Toolkit 12.1")
print("2. cuDNN")
print("3. Zaktualizuj PATH")
print("="*70 + "\n")

# Spróbuj załadować
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\nAby włączyć CUDA na RTX 4070Ti:")
        print("1. Zainstaluj CUDA Toolkit i cuDNN")
        print("2. Uaktualnij zmienne PATH")
        print("3. Restart PowerShell")
except Exception as e:
    print(f"Error: {e}")

