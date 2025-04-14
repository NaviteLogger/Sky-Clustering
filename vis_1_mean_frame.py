import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "output.csv" 

df = pd.read_csv(file_path)

width, height = 240, 80  

r = df[df.iloc[:, 0].str.startswith("r")]["mean"].values.reshape((height, width))
g = df[df.iloc[:, 0].str.startswith("g")]["mean"].values.reshape((height, width))
b = df[df.iloc[:, 0].str.startswith("b")]["mean"].values.reshape((height, width))

pixels = np.stack([r, g, b], axis=2).astype(np.uint8)

plt.imshow(pixels)
plt.axis("off")  
plt.show()
