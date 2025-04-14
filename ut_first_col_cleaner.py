import pandas as pd

file_list = []  

for filename in file_list:
    print(f"Przetwarzanie pliku: {filename}")
    
    df = pd.read_csv(filename)

    df = df.iloc[:, 1:]

    df.to_csv(filename, index=False)
    
    print(f"Nadpisano plik: {filename}")
