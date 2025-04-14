import pandas as pd

plik_summary = 'cv_summary_crop_240x60.csv'

df = pd.read_csv(plik_summary)

cvmean_max = df['CVmean'].max()
cvmean_min = df['CVmean'].min()
cvmean_mean = df['CVmean'].mean()

cvmax_max = df['CVmax'].max()
cvmax_min = df['CVmax'].min()
cvmax_mean = df['CVmax'].mean()

print(f"Statystyki dla CVmean:")
print(f"  Maksymalna: {cvmean_max}")
print(f"  Minimalna: {cvmean_min}")
print(f"  Średnia: {cvmean_mean}")

print(f"\nStatystyki dla CVmax:")
print(f"  Maksymalna: {cvmax_max}")
print(f"  Minimalna: {cvmax_min}")
print(f"  Średnia: {cvmax_mean}")
