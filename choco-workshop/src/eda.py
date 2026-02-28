import pandas
import matplotlib.pyplot as plt

CLEAN_DATA_PATH = "../data/processed/chocolate_sales.csv"
PLOT_PATH = "../reports/figures/"

clean_df = pandas.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
df = clean_df.copy()

# Basic inspection
sample = df.head()
print(sample)
sample.info(verbose=True)
sample.describe()

# Check missing values per column
for column in df.columns:
    print(f'\nColumn {column} has {df[column].isna().sum()} missing values')

# Grouping
revenue_by_country = df.groupby("country")["amount"].sum().sort_values(ascending=False)
print("\n--- Total Revenue by Country ---")
print(revenue_by_country)

revenue_by_product = df.groupby("product")["amount"].sum().sort_values(ascending=False)
print("\nTotal Revenue by Product")
print(revenue_by_product)

revenue_by_salesperson = df.groupby("sales_person")["amount"].sum().sort_values(ascending=False)
print("\nTotal Revenue by Salesperson")
print(revenue_by_salesperson)

print(df["date"])

# Aggregate by month
revenue_by_month = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().sort_index()
print("\nTotal Revenue by Month")
print(revenue_by_month)

# Plot revenue over time
x_dates = revenue_by_month.index.to_timestamp()
y_revenue = revenue_by_month.values
plt.plot(x_dates, y_revenue, marker='o', linestyle='-', color='g')
plt.title("Total Revenue by Month")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)
plt.savefig(PLOT_PATH + "revenue_by_month")
