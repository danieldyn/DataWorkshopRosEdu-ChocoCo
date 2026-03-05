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
print("\nTotal Revenue by Country")
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

# Revenue over time
plt.figure(figsize=(10, 6))
x_dates = revenue_by_month.index.to_timestamp()
y_revenue = revenue_by_month.values
plt.plot(x_dates, y_revenue, marker='o', linestyle='-', color='g')
plt.title("Total Revenue by Month")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH + "revenue_by_month.png")
plt.close()

# Top 10 products by revenue
plt.figure(figsize=(10, 6))
top_10_products = revenue_by_product.sort_values(ascending=True).head(10)
top_10_products.plot(kind="barh", color="skyblue")
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue ($)")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(PLOT_PATH + "top_10_products.png")
plt.close()

# Boxes shipped vs revenue
plt.figure(figsize=(8, 6))
plt.scatter(df["boxes_shipped"], df["amount"], alpha=0.5, color="coral")
plt.title("Boxes Shipped vs. Revenue")
plt.xlabel("Boxes Shipped")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.savefig(PLOT_PATH + "boxes_vs_revenue.png")
plt.close()
