import pandas as pd
import csv
from tqdm import tqdm

input_file = "rgb_data.csv"
output_file = "rgb_data_crop_240x60.csv"

# Liczba kolumn do zachowania
keep_columns = 43201

# Odczyt pierwszego wiersza, aby uzyskać nazwy kolumn
with open(input_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader) 
    header = header[:keep_columns] 

# Strumieniowe przetwarzanie pliku
chunk_size = 1000 

with open(output_file, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(header)
    
    with open(input_file, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.reader(f_in)
        next(reader)
        
        total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8")) - 1 
        f_in.seek(0)  
        next(reader)
        
        with tqdm(total=total_lines, desc="Przetwarzanie CSV") as pbar:
            for chunk in reader:
                writer.writerow(chunk[:keep_columns])  
                pbar.update(1)
