import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

plik_z_klastrami = "clustered_noise_vectors.csv"
folder_obrazów = "images/"

df = pd.read_csv(plik_z_klastrami)

n_samples = 12
klastry = sorted(df["Cluster"].unique())
n_rows = len(klastry)
n_cols = n_samples

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), squeeze=False)

for i, klaster in enumerate(klastry):
    cluster_images = df[df["Cluster"] == klaster]["image_name"].tolist()
    cluster_size = len(cluster_images)
    sample_files = random.sample(cluster_images, min(n_samples, cluster_size))

    for j in range(n_cols):
        ax = axes[i, j]
        ax.axis("off")

        if j < len(sample_files):
            img_path = folder_obrazów + sample_files[j]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)

    fig.text(
        0.01,
        1 - (i + 0.5) / n_rows,
        f"Klaster {klaster}\n({cluster_size} obrazków)",
        va="center",
        ha="left",
        fontsize=12
    )

plt.subplots_adjust(left=0.15, wspace=0.05, hspace=0.05)
plt.show()
