import pandas as pd
import pyarrow.parquet as pq

parquet_path = "final_data_favorita.parquet"
csv_path = "final_data_walmart.csv"

parquet_columns = pq.read_schema(parquet_path).names
csv_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

parquet_set = set(parquet_columns)
csv_set = set(csv_columns)

print("=" * 60)
print("PARQUET DATASET")
print("=" * 60)
print(f"Columns: {len(parquet_columns)}")
for col in parquet_columns:
    print(col)

print("\n" + "=" * 60)
print("CSV DATASET")
print("=" * 60)
print(f"Columns: {len(csv_columns)}")
for col in csv_columns:
    print(col)

# Σύγκριση
only_parquet = sorted(parquet_set - csv_set)
only_csv = sorted(csv_set - parquet_set)

print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

print(f"\nCols in parquet that are not in csv: {len(only_parquet)}")
for col in only_parquet:
    print(f"  - {col}")

print(f"\nCols in csv that are not in parquet: {len(only_csv)}")
for col in only_csv:
    print(f"  - {col}")

if parquet_set == csv_set:
    print("\nDatasets have same cols.")
else:
    print("\nDatasets doesnt have same cols.")