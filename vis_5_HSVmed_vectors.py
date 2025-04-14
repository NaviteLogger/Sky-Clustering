import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df_features = pd.read_csv("HSVonly_vectors.csv")

def plot_3d_scatter_multiple_views(df_features):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
    views = [(20, 30), (60, 30), (120, 30), (200, 30), (270, 30), (340, 30)]
    
    for ax, (azim, elev) in zip(axes.flatten(), views):
        ax.scatter(df_features["Hmed"], df_features["Smed"], df_features["Vmed"], 
                   c=df_features["Vmed"], cmap='viridis', alpha=0.7)
        ax.set_xlabel("Hmed")
        ax.set_ylabel("Smed")
        ax.set_zlabel("Vmed")
        ax.set_title(f"Widok (az={azim}, el={elev})")
        ax.view_init(elev, azim)
    
    plt.tight_layout()
    plt.show()

def plot_2d_scatter_pairs(df_features):
    feature_pairs = [("Hmed", "Smed"), ("Hmed", "Vmed"), ("Smed", "Vmed")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, (x_feature, y_feature) in zip(axes, feature_pairs):
        ax.scatter(df_features[x_feature], df_features[y_feature], c=df_features["Vmed"], cmap='viridis', alpha=0.7)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"{x_feature} vs {y_feature}")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_3d_scatter_multiple_views(df_features)

plot_2d_scatter_pairs(df_features)

print("Wizualizacja zakończona!")
