# SIGK - Projekt 1: Modyfikacja Obrazów
## Inpainting, Super Resolution i Denoising

Projekt realizacyjny z sieci neuronowych (SIGK) - implementacja architektur do modyfikacji obrazów przy użyciu PyTorch.

---

## 📄 Raport Projektu

Poniżej znajduje się pełny raport z projektu zawierający dokumentację eksperymentów, wyniki i wizualizacje:

### Raport: SIGK_1___final.pdf

<div align="center">

[![Pobierz raport PDF](https://img.shields.io/badge/PDF-Pobierz%20raport-red?style=for-the-badge&logo=adobe)](./data/SIGK_1___final%20%281%29.pdf)

</div>

**Raport dostępny tutaj:** [data/SIGK_1___final (1).pdf](./data/SIGK_1___final%20%281%29.pdf)

---

## Embed PDF (przeglądanie w przeglądarce)

<iframe 
  src="https://docs.google.com/gview?url=https://raw.githubusercontent.com/Michal2390/Inpainting-super-resolution-and-denoising---SIGK/main/data/SIGK_1___final%20%281%29.pdf&embedded=true" 
  style="width:100%; height:900px; border:1px solid #ccc;">
</iframe>

---

## 🚀 Szybki Start

### Wymagania
- Python 3.8+
- PyTorch z CUDA support
- GPU (opcjonalnie, ale rekomendowane)

### Instalacja

1. Klonuj repozytorium:
```bash
git clone git@github.com:Michal2390/Inpainting-super-resolution-and-denoising---SIGK.git
cd Inpainting-super-resolution-and-denoising---SIGK
```

2. Stwórz wirtualne środowisko:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

### Uruchomienie

**Windows (PowerShell):**
```powershell
.\run.ps1
```

**Linux/Mac:**
```bash
python main.py
```

---

## 📊 Struktura Projektu

```
.
├── data/                    # Zbiór danych DIV2K
│   ├── DIV2K_train_HR/     # Obrazy treningowe
│   ├── DIV2K_valid_HR/     # Obrazy walidacyjne
│   └── SIGK_1___final.pdf  # Raport projektu
├── src/                     # Kod źródłowy
│   ├── models.py           # Architektury sieci neuronowych
│   ├── train.py            # Skrypt treningowy
│   ├── evaluate.py         # Ewaluacja modeli
│   ├── dataset.py          # Przygotowanie danych
│   └── utils.py            # Funkcje pomocnicze
├── models/                  # Zapisane modele
├── results/                 # Wyniki eksperymentów
├── notebooks/               # Jupyter notebooks
│   └── evaluation.ipynb    # Analiza wyników
├── main.py                 # Główny skrypt
├── config.py               # Konfiguracja
└── requirements.txt        # Zależności Python
```

---

## 📋 Metryki Ewaluacji

Projekt wykorzystuje następujące metryki do oceny jakości:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)

---

## 📝 Zadania Projektu

Projekt obejmuje implementację 2 z 4 poniższych zadań:

### ✅ Zadanie 1: Super Resolution (Zwiększanie rozdzielczości)
- Wejście: obrazy 32×32, 64×64
- Wyjście: obrazy 256×256
- Metoda bazowa: OpenCV bicubic interpolation

### ✅ Zadanie 2: Denoising (Odszumianie)
- Szum gaussowski: σ = 0.01, 0.03
- Metoda bazowa: bilateral denoise (skimage)

### ✅ Zadanie 3: Deblurring (Deblurowanie)
- Kernel: 3×3, 5×5 (Gaussian blur)
- Metoda bazowa: Richardson-Lucy (skimage)

### ⭐ Zadanie 4: Inpainting (wypełnianie)
- Losowe wycinki: 3×3, 32×32
- Metoda bazowa: OpenCV INPAINT_TELEA

---

## 👥 Autorzy

Projekt wykonany w ramach kursu SIGK na Politechnice.

---

## 📄 Licencja

MIT License

---

## 📧 Kontakt

Aby pobrać pełny raport, kliknij: [SIGK_1___final (1).pdf](./data/SIGK_1___final%20%281%29.pdf)

