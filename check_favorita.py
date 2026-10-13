import pandas as pd

print("Φόρτωση του τελικού Parquet αρχείου...")
df = pd.read_parquet("final_data_favorita.parquet")

print("Υπολογισμός NaNs ανά στήλη...\n")
nan_counts = df.isna().sum()

# Κρατάμε μόνο τις στήλες που έχουν έστω και 1 NaN και τις ταξινομούμε φθίνουσα
missing_data = nan_counts[nan_counts > 0].sort_values(ascending=False)
total_rows = len(df)

print(f"Συνολικές γραμμές dataset: {total_rows}")
print("-" * 40)
print("ΣΤΗΛΗ".ljust(25) + "| NaNs".ljust(15) + "| ΠΟΣΟΣΤΟ %")
print("-" * 40)

if missing_data.empty:
    print("Το dataset δεν έχει απολύτως κανένα NaN!")
else:
    for col, count in missing_data.items():
        percent = (count / total_rows) * 100
        print(f"{col.ljust(25)} | {str(count).ljust(13)} | {percent:.2f}%")