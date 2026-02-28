import pandas

RAW_DATA_PATH = "../data/raw/Chocolate Sales (2).csv"
CLEAN_DATA_PATH = "../data/processed/chocolate_sales.csv"

raw_df = pandas.read_csv(RAW_DATA_PATH)
df = raw_df.copy()
initial_rows = df.index.size

# Standardise column names
df.columns = df.columns.str.lower().str.strip()
df.columns = df.columns.str.replace(" ", "_")

# Parse date column
dates_before = df["date"]
df["date"] = pandas.to_datetime(df["date"], dayfirst=True, errors="coerce")
is_nat_mask = df["date"].isna()
failed_original_dates = dates_before[is_nat_mask & dates_before.notna()]

# Convert numeric columns
df["amount"] = df["amount"].str.replace(",", "")
df["amount"] = df["amount"].str.replace("$", "")
df["boxes_shipped"] = pandas.to_numeric(df["boxes_shipped"], errors="coerce")
failed_numeric_conversion = df["boxes_shipped"].isna().sum()

# Handle missing values: remove row if amount is missing
df = df.dropna(subset=["amount"])
final_rows = df.index.size

# Remove duplicates
df = df.drop_duplicates(keep="first")

# Print conclusions to terminal
print(f'Initial number of rows: {initial_rows}')
print(f'Final number or rows: {final_rows}')
print(f'Failed date conversion: {failed_original_dates}')
print(f'Failed numeric conversion: {failed_numeric_conversion}')

# Write clean data
df.to_csv(CLEAN_DATA_PATH, index=False)
