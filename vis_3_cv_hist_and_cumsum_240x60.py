import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = "cv_summary_cropped_240x60.csv"

bins = np.arange(0, 1.1, 0.1)
bin_labels = [f"{round(bins[i], 1)} - {round(bins[i+1], 1)}" for i in range(len(bins)-1)]

def count_intervals(file):
    df = pd.read_csv(file, usecols=[1, 2], skiprows=1, header=None).astype(float)

    counts_col1, _ = np.histogram(df.iloc[:, 0], bins=bins)
    counts_col2, _ = np.histogram(df.iloc[:, 1], bins=bins)

    cumulative_col1 = np.cumsum(counts_col1[::-1])[::-1]
    cumulative_col2 = np.cumsum(counts_col2[::-1])[::-1]

    df_counts = pd.DataFrame({
        "Przedział": bin_labels,
        "CVmean": counts_col1,
        "CVmax": counts_col2
    })

    df_cumulative = pd.DataFrame({
        "Powyżej": [f">{round(b, 1)}" for b in bins[:-1]],
        "CVmean (kumul.)": cumulative_col1,
        "CVmax (kumul.)": cumulative_col2
    })

    return df_counts, df_cumulative

def plot_tables(df_counts, df_cumulative):
    
    plt.figure(figsize=(10, 5))
    plt.bar(df_counts["Przedział"], df_counts["CVmean"], label="CVmean", alpha=0.7, color="blue")
    plt.bar(df_counts["Przedział"], df_counts["CVmax"], label="CVmax", alpha=0.7, color="orange")
    plt.xticks(rotation=45)
    plt.xlabel("Przedział wartości")
    plt.ylabel("Liczba wystąpień")
    plt.title("Histogram wartości CVmean i CVmax")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df_cumulative["Powyżej"], df_cumulative["CVmean (kumul.)"], marker="o", label="CVmean (kumul.)", color="blue")
    plt.plot(df_cumulative["Powyżej"], df_cumulative["CVmax (kumul.)"], marker="o", label="CVmax (kumul.)", color="orange")
    plt.xticks(rotation=45)
    plt.xlabel("Próg wartości")
    plt.ylabel("Kumulatywna liczba wystąpień")
    plt.title("Kumulatywna suma wartości CVmean i CVmax")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

table_counts, table_cumulative = count_intervals(input_file)

print("Tabela zliczeń:")
print(table_counts)
print("\nTabela kumulatywna:")
print(table_cumulative)

plot_tables(table_counts, table_cumulative)
