import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def compute_elbow_metrics(data, k_values):
    inertia = []
    silhouette_scores = []
    elbow_scores = []
    
    print("Obliczanie metryk dla różnych wartości k...")
    for k in tqdm(k_values, desc="Analiza klastrów"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data.iloc[:, 1:]) 
        inertia.append(kmeans.inertia_)
        
        if k > 1:
            silhouette = silhouette_score(data.iloc[:, 1:], labels)
            silhouette_scores.append(silhouette)
        else:
            silhouette_scores.append(np.nan)
    
    # Wyliczanie drugiej różnicy inercji (przybliżona druga pochodna)
    elbow_scores = np.gradient(np.gradient(inertia))
    suggested_k = k_values[np.argmin(elbow_scores)]
    
    return inertia, silhouette_scores, elbow_scores, suggested_k

filenames = ["HSVonly_vectors.csv"]  
k_values = range(2, 15)

for filename in filenames:
    print(f"\n🔍 Przetwarzanie pliku: {filename}")
    df = pd.read_csv(filename)
    image_names = df.iloc[:, 0] 
    data = df.iloc[:, 1:]
    
    inertia, silhouette_scores, elbow_scores, suggested_k = compute_elbow_metrics(df, k_values)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(k_values, inertia, marker="o", linestyle="-", label="Inercja", color='blue')
    ax2.plot(k_values, silhouette_scores, marker="s", linestyle="--", label="Silhouette Score", color='green')
    ax1.set_xlabel("Liczba klastrów (k)")
    ax1.set_ylabel("Inercja", color='blue')
    ax2.set_ylabel("Silhouette Score", color='green')
    plt.title(f"Analiza optymalnej liczby klastrów - {filename}")
    ax1.axvline(suggested_k, color='red', linestyle='dotted', label=f"Sugerowane k={suggested_k}")
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    plt.show()
    
    optimal_k = int(input(f"Podaj wybraną liczbę klastrów dla {filename} (sugerowane: {suggested_k}): "))
    
    print(f"Trwa klastrowanie dla k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(data)
    
    df.insert(0, "Image_Name", image_names)
    
    output_filename = f"clustered_{filename}"
    df.to_csv(output_filename, index=False)
    print(f"✅ Plik z klastrami zapisany jako '{output_filename}'.")
