import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_3d_clusters(data, labels, title="Wizualizacja klastrów DBSCAN"):

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))  

    legend_entries = []  
    legend_labels = []  

    for label in unique_labels:
        cluster_points = data[labels == label]
        cluster_size = len(cluster_points)

        if label == -1:
            color = "black"  
            label_name = f"Szum (-1), {cluster_size} pkt."
        else:
            color = colors(label)
            label_name = f"Cluster {label}, {cluster_size} pkt."

        scatter1 = ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                               color=color, alpha=0.6, edgecolors='k')

        if label != -1:
            scatter2 = ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                                   color=color, alpha=0.6, edgecolors='k')

        legend_entries.append(scatter1)
        legend_labels.append(label_name)

    for ax in [ax1, ax2]:
        ax.set_xlabel("Mediana Hue (Hmed)")
        ax.set_ylabel("Mediana Saturation (Smed)")
        ax.set_zlabel("Mediana Value (Vmed)")

    ax1.legend(legend_entries, legend_labels, loc="upper left")
    ax2.legend(legend_entries[1:], legend_labels[1:], loc="upper left")

    ax1.set_title("Wszystkie punkty (w tym szum)")
    ax2.set_title("Tylko rzeczywiste klastry")
    
    plt.tight_layout()
    plt.show()

filename = "dbscan_0.008_825_HSVonly_vectors.csv"

print(f"\n🔍 Wczytywanie pliku: {filename}")
df = pd.read_csv(filename)

expected_columns = 5
if df.shape[1] != expected_columns:
    raise ValueError(f"Plik powinien mieć {expected_columns} kolumn, a ma {df.shape[1]}!")

data = df.iloc[:, 1:4].values
labels = df.iloc[:, 4].values

plot_3d_clusters(data, labels)
