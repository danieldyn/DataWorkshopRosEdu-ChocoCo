import numpy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from train import X_train, X_test, y_train, y_test, df, sorted_indices, split_idx

final_model = RandomForestRegressor(n_estimators=200, random_state=42)
final_model.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = final_model.predict(X_test)

# Metrics
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = numpy.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print(f'MAE:  ${test_mae:.2f}')
print(f'RMSE: ${test_rmse:.2f}')
print(f'R^2:   {test_r2:.4f}')

# Error analysis
print("\nError Analysis")

# Reconstruct the original dataframe
df_test_original = df.loc[sorted_indices].iloc[split_idx:].copy()

# Add our predictions and calculate the absolute error for every single row
df_test_original["predicted_amount"] = y_pred
df_test_original["abs_error"] = numpy.abs(df_test_original["amount"] - df_test_original["predicted_amount"])

# Group by country to find the least accurate
error_by_country = df_test_original.groupby("country")["abs_error"].mean().sort_values(ascending=False)
print("\nAverage Error by Country:")
print(error_by_country)

# Group by product to find worst predictions
error_by_product = df_test_original.groupby("product")["abs_error"].mean().sort_values(ascending=False)
print("\nAverage Error by Product (Top 10 Worst):")
print(error_by_product.head(10))
