import csv
from tqdm import tqdm
import re 

def keep_columns(input_file, output_file, columns_to_keep):
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        keep_columns = [i for i in range(len(header)) if i in columns_to_keep]
        new_header = [header[i] for i in keep_columns]
        
    with open(output_file, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(new_header)
        
        with open(input_file, "r", newline="", encoding="utf-8") as f_in:
            reader = csv.reader(f_in)
            next(reader)  
            
            total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8")) - 1
            f_in.seek(0)
            next(reader)  
            
            with tqdm(total=total_lines, desc="Przetwarzanie CSV") as pbar:
                for row in reader:
                    new_row = [row[i] for i in keep_columns]
                    writer.writerow(new_row)
                    pbar.update(1)

input_filename = "cv_summary_cropped_240x60_treshold_20.csv"

px_values = []

with open(input_filename, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    
    for row in reader:
        try:
            match = re.search(r'\d+', row['px_id'])
            if match:
                px_values.append(int(match.group()))
        except (ValueError, KeyError):
            pass

keep = [0]
for value in px_values:
    keep.extend([3*value, 3*value-1, 3*value-2])

keep_columns("rgb_data_cropped_240x60.csv", "rgb_data_cropped_240x60_treshold_20.csv", keep)
