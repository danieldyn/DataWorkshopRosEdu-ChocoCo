import pandas

CLEAN_DATA_PATH = "../data/processed/chocolate_sales.csv"

clean_df = pandas.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
df = clean_df.copy()

# Create date-derived features, including weekend bonus check
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Choose Amount as prediction target
y = df["amount"]

# Decide which columns are inputs
feature_columns = ["boxes_shipped", "country", "product", "sales_person", "month", "day_of_week", "is_weekend"]
X = df[feature_columns]

# Encode categorical variables
X_encoded = pandas.get_dummies(X, columns=["country", "product", "sales_person"], dtype=int)

print("Original Features")
print(X.head())
print("\nEncoded Features")
print(X_encoded.head())
