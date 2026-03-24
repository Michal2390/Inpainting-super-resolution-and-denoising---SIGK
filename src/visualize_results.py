import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from src.models import create_model
from src.dataset import DIV2KDataset


def save_comparison_plot(original, distorted, restored, task, index, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Oryginał (HR/Clean)',
              'Zniekształcony (LR/Noisy)',
              f'Wynik Modelu ({task.upper()})']

    for ax, img, title in zip(axes, [original, distorted, restored], titles):
        # Konwersja z Tensor (C, H, W) na numpy (H, W, C)
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()

        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{task}_comparison_{index}.png'))
    plt.close()


def run_visualization():
    device = 'cuda' #if torch.cuda.is_available() else 'cpu'
    data_dir = '../data/DIV2K_valid_HR/'
    output_path = 'results/plots/'
    os.makedirs(output_path, exist_ok=True)

    tasks = {
        'sr': {'path': '../models/sr/best_sr_model.pth', 'scale': 4, 'sigma': 0.1},
        'denoise': {'path': '../models/denoise/best_denoise_model.pth', 'scale': 2, 'sigma': 0.1},
        'inpainting': {'path': '../models/inpainting/best_inpainting_model.pth', 'scale': 1, 'sigma': 0}
    }

    for task, params in tasks.items():
        print(f"Generowanie wizualizacji dla: {task}")

        # Ładowanie modelu
        model = create_model(task=task, scale_factor=params['scale'], device=device)
        checkpoint = torch.load(params['path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Zbiór danych (bez augmentacji)
        dataset = DIV2KDataset(data_dir, task=task, scale_factor=params['scale'],
                               sigma=params['sigma'], patch_size=256, augment=False)

        with torch.no_grad():
            for i in range(3):  # Generujemy 3 przykłady dla każdego zadania
                batch = dataset[i]

                if task == 'sr':
                    input_img = batch['lr'].unsqueeze(0).to(device).float()
                    target_img = batch['hr']
                    distorted_img = batch['lr']
                elif task == 'denoise':
                    input_img = batch['noisy'].unsqueeze(0).to(device).float()
                    target_img = batch['clean']
                    distorted_img = batch['noisy']
                elif task == 'inpainting':
                    input_img = batch['masked'].unsqueeze(0).to(device).float()
                    target_img = batch['clean']
                    distorted_img = batch['masked']

                output = model(input_img).squeeze(0).cpu()

                save_comparison_plot(target_img, distorted_img, output, task, i, output_path)


if __name__ == '__main__':
    run_visualization()
    print("Wizualizacje zostały zapisane w folderze results/plots/")