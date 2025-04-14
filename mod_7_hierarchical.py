import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from mpl_toolkits.mplot3d import Axes3D

CLUSTER_COLORS = {
    1: "#FF7700",  
    2: "#AA00FF", 
    3: "#00C8C8",  
    4: "#A4DE02", 
    5: "#FF1493"   
}

def plot_3d_clusters(data, labels):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = data[labels == label]
        color = CLUSTER_COLORS.get(label, "#000000")
        cluster_size = len(cluster_points)
        label_name = f"Cluster {label} ({cluster_size} pkt)"

        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   label=label_name, color=color, alpha=0.8, edgecolors='k')

    ax.set_xlabel("Mediana Hue (Hmed)")
    ax.set_ylabel("Mediana Saturation (Smed)")
    ax.set_zlabel("Mediana Value (Vmed)")
    ax.legend(loc="upper left")

    ax.set_title("Klastrowanie metodą Warda (5 klastrów)")

    plt.tight_layout()
    plt.show()

filename = "HSVonly_vectors.csv"
print(f"\n🔍 Przetwarzanie pliku: {filename}")
df = pd.read_csv(filename)
image_names = df.iloc[:, 0]
data = df.iloc[:, 1:].values

linkage_matrix = linkage(data, method='ward')

n_clusters = 5
labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

df["Cluster"] = labels

plot_3d_clusters(data, labels)

output_filename = f"ward_{n_clusters}_{filename}"
df.to_csv(output_filename, index=False)
print(f"✅ Plik z klastrami zapisany jako '{output_filename}'.")
