import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

def find_optimal_eps(data, k=5):
    
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    
    distances = np.sort(distances[:, k-1])
    
    plt.figure(figsize=(6, 4))
    plt.plot(distances)
    plt.xlabel("Próbki")
    plt.ylabel(f"{k}-NN distance")
    plt.title("Wykres k-NN (wybór eps)")
    plt.grid()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_3d_clusters(data, labels, title="Wizualizacja klastrów DBSCAN"):

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for label in unique_labels:
        if label == -1:
            color = "black" 
            label_name = "Szum (-1)"
        else:
            color = colors(label)
            label_name = f"Cluster {label}"
        
        cluster_points = data[labels == label]

        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                    label=label_name, color=color, alpha=0.6, edgecolors='k')

        if label != -1:
            ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                        label=label_name, color=color, alpha=0.6, edgecolors='k')

    for ax in [ax1, ax2]:
        ax.set_xlabel("Mediana Hue (Hmed)")
        ax.set_ylabel("Mediana Saturation (Smed)")
        ax.set_zlabel("Mediana Value (Vmed)")
        ax.legend(loc="upper left")

    ax1.set_title("Wszystkie punkty (w tym szum)")
    ax2.set_title("Tylko rzeczywiste klastry")
    
    plt.tight_layout()
    plt.show()


filename = "HSVonly_vectors.csv"  

print(f"\n🔍 Przetwarzanie pliku: {filename}")
df = pd.read_csv(filename)
image_names = df.iloc[:, 0]
data = df.iloc[:, 1:].values  

find_optimal_eps(data)

eps_value = float(input("Podaj wartość eps dla DBSCAN: "))
min_samples_value = int(input("Podaj wartość min_samples: "))

print(f"Trwa klastrowanie DBSCAN (eps={eps_value}, min_samples={min_samples_value})...")
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
labels = dbscan.fit_predict(data)
df["Cluster"] = labels

plot_3d_clusters(data, labels)

output_filename = f"dbscan_{eps_value}_{min_samples_value}_{filename}"
df.to_csv(output_filename, index=False)
print(f"✅ Plik z klastrami zapisany jako '{output_filename}'.")
