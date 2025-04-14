import pandas as pd
import numpy as np
import colorsys
from tqdm import tqdm  
import matplotlib.pyplot as plt

df = pd.read_csv("rgb_data_cropped_240x60_treshold_20.csv")

image_names = df["image_name"]
df = df.drop(columns=["image_name"])

def calculate_features(df):
    features = {}
    
    r = df.iloc[:, 0::3].values / 255.0  
    g = df.iloc[:, 1::3].values / 255.0
    b = df.iloc[:, 2::3].values / 255.0
    
    v = (r + g + b) / 3
    features["Vavg"] = np.mean(v, axis=1)
    features["Vstd"] = np.std(v, axis=1)
    features["Vmed"] = np.median(v, axis=1)
    features["IQRV"] = np.percentile(v, 75, axis=1) - np.percentile(v, 25, axis=1)
    features["P90V"] = np.percentile(v, 90, axis=1)
    features["Vmax_Vmin"] = np.max(v, axis=1) - np.min(v, axis=1)
    
    features["IQRRGB"] = (
        np.percentile(r, 75, axis=1) - np.percentile(r, 25, axis=1) +
        np.percentile(g, 75, axis=1) - np.percentile(g, 25, axis=1) +
        np.percentile(b, 75, axis=1) - np.percentile(b, 25, axis=1)
    ) / 3
    
    features["GR"] = np.mean(g / (r + 1e-6), axis=1)
    features["GB"] = np.mean(g / (b + 1e-6), axis=1)
    
    h_values, s_values, s_std_values = [], [], []
    for i in tqdm(range(r.shape[0]), desc="Przetwarzanie HSV"):
        hsv_pixels = [colorsys.rgb_to_hsv(r[i, j], g[i, j], b[i, j]) for j in range(r.shape[1])]
        h, s, _ = zip(*hsv_pixels)
        h_values.append(np.mean(h))
        s_values.append(np.mean(s))
        s_std_values.append(np.std(s))
    
    features["Savg"] = s_values
    features["Sstd"] = s_std_values
    features["Havg"] = h_values
    
    df_features = pd.DataFrame(features)

    df_features["GR"] = df_features["GR"] / (df_features["GR"].max() + 1e-6)
    df_features["GB"] = df_features["GB"] / (df_features["GB"].max() + 1e-6)
    
    return df_features

def plot_all_histograms(df_features):
    features = df_features.drop(columns=["image_name"]).columns

    num_features = len(features)
    
    rows = (num_features // 3) + (num_features % 3 > 0)
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(df_features[feature], bins=30, color='blue', alpha=0.7)
        ax.set_title(f"Histogram - {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Częstotliwość")
        ax.grid(True)
        #ax.set_yscale('log')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout() 
    plt.show()


print("Obliczanie cech...")
df_features = calculate_features(df)
df_features.insert(0, "image_name", image_names)

plot_all_histograms(df_features)

noise_features = df_features[["image_name", "Vavg", "Vstd", "IQRRGB", "GR", "GB", "Savg", "Havg"]]
# cloud_features = df_features[["image_name", "Vmed", "Sstd", "IQRV", "Savg", "Vstd"]]
# day_night_features = df_features[["image_name", "Vavg", "Vmed", "Savg", "P90V", "Vmax_Vmin"]]

# Zapisujemy wyniki
print("Zapisywanie wyników...")
noise_features.to_csv("noise_vectors.csv", index=False)
# cloud_features.to_csv("cloud_vectors.csv", index=False)
# day_night_features.to_csv("day_night_vectors.csv", index=False)

print("Wektory zapisane do plików CSV!")
