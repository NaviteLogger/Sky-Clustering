import pandas as pd
import numpy as np
import colorsys
from tqdm import tqdm  
import matplotlib.pyplot as plt

df = pd.read_csv("rgb_data_cropped_240x60_treshold_20.csv")

image_names = df["image_name"]
df = df.drop(columns=["image_name"]) 

def calculate_selected_features(df):
    features = {}
    
    r = df.iloc[:, 0::3].values / 255.0  
    g = df.iloc[:, 1::3].values / 255.0
    b = df.iloc[:, 2::3].values / 255.0
    
    h_values, s_values, v_values = [], [], []
    for i in tqdm(range(r.shape[0]), desc="Przetwarzanie HSV"):
        hsv_pixels = [colorsys.rgb_to_hsv(r[i, j], g[i, j], b[i, j]) for j in range(r.shape[1])]
        h, s, v = zip(*hsv_pixels)
        h_values.append((np.mean(h), np.median(h)))
        s_values.append((np.mean(s), np.median(s)))
        v_values.append((np.mean(v), np.median(v)))
    
    features["Havg"], features["Hmed"] = zip(*h_values)
    features["Savg"], features["Smed"] = zip(*s_values)
    features["Vavg"], features["Vmed"] = zip(*v_values)
    
    return pd.DataFrame(features)


def plot_histograms(df_features):
    features = df_features.columns[1:]
    num_features = len(features)
    rows, cols = 3, 2 

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(df_features[feature], bins=30, color='blue', alpha=0.7)
        ax.set_title(f"Histogram - {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Częstotliwość")
        ax.grid(True)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

df_features = calculate_selected_features(df)
df_features.insert(0, "image_name", image_names)

plot_histograms(df_features)

hsv_med_only = df_features[["image_name", "Hmed", "Smed", "Vmed"]]

hsv_med_only.to_csv("HSVonly_vectors.csv", index=False)

print("Wektory cech zapisane do HSVonly_vectors.csv!")
