import pandas as pd

PARQUET_PATH = r"../../corporaction_favorita_dataset/final_data_favorita.parquet"

# Load ONLY the specific columns needed to save RAM
cols_to_load = [
    'store_nbr', 'store_type', 'cluster', 'Is_Christmas_Season',
    'IsHoliday', 'weather_condition', 'city', 'unit_sales'
]

print(f">>> Loading specific columns from Parquet...\n")
df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)


print("="*60)
print("1. SALES AND ROWS DISTRIBUTION BY STORE TYPE")
print("="*60)
type_stats = df.groupby('store_type')['unit_sales'].agg(['count', 'mean']).rename(columns={'count': 'Total_Rows', 'mean': 'Avg_Sales'})
print(type_stats.sort_values(by='Total_Rows', ascending=False).to_string())


print("\n" + "="*60)
print("2. STORE TYPE vs CLUSTER DISTRIBUTION (Which types belong to which clusters)")
print("="*60)
# Crosstab creates a two-way table (rows: store_type, columns: cluster)
type_cluster = pd.crosstab(df['store_type'], df['cluster'])
print(type_cluster.to_string())


print("\n" + "="*60)
print("3. DISTRIBUTION AND SALES: IS_CHRISTMAS_SEASON")
print("="*60)
xmas_stats = df.groupby('Is_Christmas_Season')['unit_sales'].agg(['count', 'mean']).rename(columns={'count': 'Total_Rows', 'mean': 'Avg_Sales'})
print(xmas_stats.to_string())


print("\n" + "="*60)
print("4. DISTRIBUTION AND SALES: IS_HOLIDAY")
print("="*60)
hol_stats = df.groupby('IsHoliday')['unit_sales'].agg(['count', 'mean']).rename(columns={'count': 'Total_Rows', 'mean': 'Avg_Sales'})
print(hol_stats.to_string())


print("\n" + "="*60)
print("5. DISTRIBUTION AND SALES BY WEATHER CONDITION")
print("="*60)
weather_stats = df.groupby('weather_condition')['unit_sales'].agg(['count', 'mean']).rename(columns={'count': 'Total_Rows', 'mean': 'Avg_Sales'})
print(weather_stats.sort_values(by='Total_Rows', ascending=False).to_string())


print("\n" + "="*60)
print("6. DISTRIBUTION AND SALES BY CITY")
print("="*60)
city_stats = df.groupby('city')['unit_sales'].agg(['count', 'mean']).rename(columns={'count': 'Total_Rows', 'mean': 'Avg_Sales'})
print(city_stats.sort_values(by='Total_Rows', ascending=False).to_string())


print("\n" + "="*60)
print("7. DISTRIBUTION AND SALES BY STORE NUMBER")
print("="*60)
# Sorting by store_nbr to easily read from 1 to 54
store_stats = df.groupby('store_nbr')['unit_sales'].agg(['count', 'mean']).rename(columns={'count': 'Total_Rows', 'mean': 'Avg_Sales'})
print(store_stats.sort_index().to_string())
print("\n" + "="*60)