import csv

input_filename = "cv_summary_cropped_240x60.csv"
output_filename = "cv_summary_cropped_240x60_treshold_20.csv"

threshold = 0.2

# Wczytanie i filtrowanie danych
with open(input_filename, newline='', encoding='utf-8') as infile, open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    header = next(reader, None)
    if header:
        writer.writerow(header)
    
    for row in reader:
        try:
            if float(row[1]) <= threshold:
                writer.writerow(row)
        except (ValueError, IndexError):
            pass

print(f"Przetwarzanie zakończone. Wynik zapisano w {output_filename}")
