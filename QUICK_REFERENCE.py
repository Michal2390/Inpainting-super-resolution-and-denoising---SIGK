#!/usr/bin/env python3
"""
QUICK REFERENCE - Projekt SIGK
Szybka ściągawka do uruchamiania projektu
"""

# ════════════════════════════════════════════════════════════════════════════════
# 🚀 SZYBKI START - KOPIA WKLEJ
# ════════════════════════════════════════════════════════════════════════════════

"""
KROK 1: Instalacja bibliotek
────────────────────────────
python -m pip install -r requirements.txt


KROK 2: Test weryfikacyjny
──────────────────────────
python quickstart.py


KROK 3: Trenowanie (duży plik - ~25GB data)
─────────────────────────────────────────
python main.py


KROK 4: Ewaluacja
───────────────
jupyter notebook notebooks/evaluation.ipynb
"""

# ════════════════════════════════════════════════════════════════════════════════
# 📚 DOKUMENTACJA
# ════════════════════════════════════════════════════════════════════════════════

"""
GŁÓWNE PLIKI DOKUMENTACJI:
─────────────────────────
1. README.md - Główna dokumentacja
2. INSTRUKCJA.md - Szczegółowe instrukcje (PRZECZYTAJ!)
3. PYCHARM_SETUP.md - Konfiguracja PyCharm
4. EXPERIMENTS.md - Dokumentacja eksperymentów
5. SUMMARY.md - Podsumowanie projektu (TUTAJ JESTEŚ)

GDZIE SZUKAĆ POMOCY:
──────────────────
❌ Coś nie działa?  → INSTRUKCJA.md sekcja "Rozwiązywanie Problemów"
❌ Jak uruchomić?   → INSTRUKCJA.md sekcja "Szybki Start"
❌ PyCharm problemy? → PYCHARM_SETUP.md
"""

# ════════════════════════════════════════════════════════════════════════════════
# 💻 KODY DO SZYBKIEGO TESTOWANIA
# ════════════════════════════════════════════════════════════════════════════════

"""
TEST 1: Sprawdzić Python
──────────────────────
python --version
→ Powinno być 3.8 lub wyżej


TEST 2: Sprawdzić PyTorch
─────────────────────────
python -c "import torch; print(f'PyTorch: {torch.__version__}'); 
           print(f'GPU: {torch.cuda.is_available()}')"


TEST 3: Sprawdzić OpenCV
────────────────────────
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"


TEST 4: Stwórz model (bez trenowania)
──────────────────────────────────────
python -c "
import torch
from src.models import create_model

model = create_model(task='sr', scale_factor=4, device='cpu')
print(f'Model created: {sum(p.numel() for p in model.parameters()):,} parameters')
"


TEST 5: Test dataset
────────────────────
python -c "
from src.dataset import DIV2KDataset

# Jeśli data directory nie istnieje, stworzysz folder
# dataset = DIV2KDataset('data/DIV2K_train_HR/', task='sr')
# print(f'Dataset size: {len(dataset)}')
print('Dataset code ready!')
"
"""

# ════════════════════════════════════════════════════════════════════════════════
# 🎯 ARGUMENTY TRENOWANIA
# ════════════════════════════════════════════════════════════════════════════════

"""
KONFIGURACJA W config.py:
─────────────────────────

TRAINING_CONFIG = {
    'batch_size': 32,      # Zmniejsz na 8 jeśli mało RAM
    'learning_rate': 1e-3,
    'num_epochs': 50,
    'device': 'cuda',      # Zmień na 'cpu' jeśli brak GPU
}

SR_CONFIG = {
    'scale_factor': 4,     # 4x upscaling
}

DENOISE_CONFIG = {
    'sigma': 0.1,          # Siła szumu
}
"""

# ════════════════════════════════════════════════════════════════════════════════
# 📊 STRUKTURA DANYCH
# ════════════════════════════════════════════════════════════════════════════════

"""
Po uruchomieniu main.py utworzą się:

data/
  ├─ DIV2K_train_HR/      (800 obrazów - pobrać auto)
  └─ DIV2K_valid_HR/      (100 obrazów - pobrać auto)

models/
  ├─ sr/
  │   ├─ best_sr_model.pth       (najlepszy checkpoint SR)
  │   ├─ sr_model_epoch_10.pth   (checkpoint epoch 10)
  │   ├─ sr_model_epoch_20.pth   (checkpoint epoch 20)
  │   └─ history.json            (historia trenowania)
  └─ denoise/
      ├─ best_denoise_model.pth
      ├─ denoise_model_epoch_*.pth
      └─ history.json

results/
  ├─ summary_results.csv          (tabela wyników)
  ├─ sr_comparison.png            (porównanie SR)
  ├─ denoise_comparison.png       (porównanie denoise)
  ├─ metric_distributions.png     (rozkłady metryk)
  └─ training_history.png         (krzywe trenowania)
"""

# ════════════════════════════════════════════════════════════════════════════════
# 🔧 ZAAWANSOWANE - RĘCZNE TRENOWANIE
# ════════════════════════════════════════════════════════════════════════════════

"""
Jeśli chcesz trenować ręcznie:

import torch
from src.train import train_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Super-Resolution
trainer_sr, history_sr = train_model(
    task='sr',
    scale_factor=4,
    num_epochs=50,
    device=device,
    checkpoint_dir='models/'
)

# Denoising
trainer_dn, history_dn = train_model(
    task='denoise',
    sigma=0.1,
    num_epochs=50,
    device=device,
    checkpoint_dir='models/'
)
"""

# ════════════════════════════════════════════════════════════════════════════════
# 📈 MONITORING TRENOWANIA
# ════════════════════════════════════════════════════════════════════════════════

"""
TensorBoard:
────────────
1. Uruchom trenowanie w jednym terminalu:
   python main.py

2. W innym terminalu:
   tensorboard --logdir=models/sr/logs/
   
3. Otwórz przeglądarkę:
   http://localhost:6006/

Zobaczysz:
- Krzywe straty (Loss)
- PSNR i SSIM na validacji
- Inne metryki
"""

# ════════════════════════════════════════════════════════════════════════════════
# 🎨 JUPYTER NOTEBOOK
# ════════════════════════════════════════════════════════════════════════════════

"""
Ewaluacja w notebooku:
──────────────────────
jupyter notebook notebooks/evaluation.ipynb

Notebook zawiera:
1. Ładowanie modeli
2. Ewaluację na zbiorze testowym
3. Porównanie z baseline'ami
4. Wizualizacje
5. Tabele wyników
6. Wykresy metryk
"""

# ════════════════════════════════════════════════════════════════════════════════
# ⚡ SZYBKIE PORADY
# ════════════════════════════════════════════════════════════════════════════════

"""
TIP 1: GPU vs CPU
──────────────────
GPU: ~6 godzin trenowania
CPU: ~30 godzin trenowania

TIP 2: Brakuje RAM?
─────────────────
Zmniejsz batch_size w config.py:
'batch_size': 8  # Zamiast 32

TIP 3: Dataset duży?
────────────────────
DIV2K = ~25GB
Pobierze się automatycznie przy main.py

TIP 4: Trenowanie przerwane?
────────────────────────────
Checkpoint'y zapisane automatycznie
Możesz wznowić trenowanie

TIP 5: Chcesz eksperymentować?
──────────────────────────────
Zmodyfikuj model w src/models.py
Zmień hyperparametry w config.py
"""

# ════════════════════════════════════════════════════════════════════════════════
# 🚨 ROZWIĄZYWANIE PROBLEMÓW
# ════════════════════════════════════════════════════════════════════════════════

"""
ERROR 1: "ModuleNotFoundError: No module named 'torch'"
────────────────────────────────────────────────────
ROZWIĄZANIE:
python -m pip install torch torchvision


ERROR 2: "CUDA out of memory"
───────────────────────────
ROZWIĄZANIE:
- Zmniejsz batch_size (32 → 8)
- Zmniejsz patch_size (256 → 128)
- Użyj CPU zamiast GPU


ERROR 3: "DIV2K dataset not found"
─────────────────────────────────
ROZWIĄZANIE:
- main.py pobierze dane automatycznie
- Jeśli problem, pobierz ręcznie z:
  https://data.vision.ee.ethz.ch/cvl/DIV2K/


ERROR 4: "Jupyter notebook not found"
──────────────────────────────────
ROZWIĄZANIE:
python -m pip install jupyter
jupyter notebook notebooks/evaluation.ipynb


ERROR 5: "Python command not found"
──────────────────────────────────
ROZWIĄZANIE:
- Zamiast: python
- Spróbuj: py
- Lub: <full_path_to_python>/python.exe
"""

# ════════════════════════════════════════════════════════════════════════════════
# 📞 PRZYDATNE LINKI
# ════════════════════════════════════════════════════════════════════════════════

"""
DIV2K Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
PyTorch: https://pytorch.org/
OpenCV: https://opencv.org/
scikit-image: https://scikit-image.org/
TensorBoard: https://www.tensorflow.org/tensorboard
Jupyter: https://jupyter.org/
"""

# ════════════════════════════════════════════════════════════════════════════════
# ✅ CHECKLIST - ZANIM ODDASZ PROJEKT
# ════════════════════════════════════════════════════════════════════════════════

"""
□ Python zainstalowany (3.8+)
□ Zależności zainstalowane (pip install -r requirements.txt)
□ quickstart.py przeszedł testy
□ main.py uruchomiony i trenowanie powiodło się
□ evaluation.ipynb wygenerował wyniki
□ results/ folder zawiera porównania i wykresy
□ Przeczytano INSTRUKCJA.md
□ Przeczytano EXPERIMENTS.md
□ Dokumentacja eksperymentów uzupełniona
□ Modele zapisane w models/
□ Tabele metryki wygenerowane
□ Projekt gotowy do prezentacji
"""

# ════════════════════════════════════════════════════════════════════════════════

print(__doc__)

# EOF

