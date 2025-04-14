import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cv_file = "output.csv"
image_size = (80, 240) 

df = pd.read_csv(cv_file)

pixels = len(df) // 3  
cv_values = df['cv'].values.reshape(pixels, 3)  
cv_mean = np.mean(cv_values, axis=1).reshape(image_size)  
cv_max = np.max(cv_values, axis=1).reshape(image_size)  

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)
cmap = ax.imshow(cv_mean, cmap="coolwarm")
ax.set_title("Interaktywna wizualizacja CV")
ax.axis("off")
cbar = plt.colorbar(cmap, ax=ax)
cbar.set_label("CV")

def on_hover(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            mean_value = cv_mean[y, x]
            max_value = cv_max[y, x]
            ax.set_title(f"Średnia CV: {mean_value:.3f}, Maksymalne CV: {max_value:.3f}")
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)
plt.show()
