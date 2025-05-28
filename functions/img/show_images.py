from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


def show_images(df_extremes, folder_path, ncols=3):
    
    folder = Path(folder_path)
    n = len(df_extremes)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for idx, row in df_extremes.iterrows():
        img_path = Path(row["filename"])
        with Image.open(img_path) as img:
            axes[idx].imshow(img)
            axes[idx].axis('off')

    # Cacher les cases vides si besoin
    for i in range(len(df_extremes), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()