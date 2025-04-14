import pandas as pd
import numpy as np
from PIL import Image
import os

input_file = "rgb_data_cropped_240x60_treshold_20.csv"
image_folder = "images/"  

width, height = 240, 60
num_pixels = width * height  

def row_to_image(row_index, output_file="output.png"):
    
    df_header = pd.read_csv(input_file, nrows=0)
    
    df = pd.read_csv(input_file, skiprows=row_index, nrows=1, header=None)

    original_image_name = df.iloc[0, 0]

    original_image_path = os.path.join(image_folder, original_image_name)

    if os.path.exists(original_image_path):
        original_img = Image.open(original_image_path)
    else:
        print(f"Ostrzeżenie: Nie znaleziono {original_image_path}, tworzenie pustego obrazu.")
        original_img = Image.new("RGB", (width, height), (128, 128, 128)) 

    pixels = np.zeros((height, width, 3), dtype=np.uint8)

    for col_index in range(1, len(df.columns), 3):
        r_col_name = df_header.columns[col_index] 
        pixel_index = int(r_col_name[1:])  

        y = pixel_index // width
        x = pixel_index % width

        pixels[y, x] = [df.iloc[0, col_index], df.iloc[0, col_index + 1], df.iloc[0, col_index + 2]]

    filtered_img = Image.fromarray(pixels, "RGB")

    combined_width = width * 2
    combined_img = Image.new("RGB", (combined_width, height))
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(filtered_img, (width, 0))

    combined_img.save(output_file)
    # combined_img.show()

#for n in range(1, 44414, 1):
    #print(n)
    #row_to_image(n, f"sky_vision/image_{n}.png")
row_to_image(44273, f"sky_vision/test.png")