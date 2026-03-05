import numpy
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from features import df, X_encoded, y

# Sort by date
sorted_indices = df.sort_values("date").index

# Apply order to features and target
X_sorted = X_encoded.loc[sorted_indices]
y_sorted = y.loc[sorted_indices]

# Cutoff index
split_idx = int(len(X_sorted) * 0.8)

# Split data
X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

print(f'Training rows: {len(X_train)} | Testing rows: {len(X_test)}')

# Predict the mean and median of the training set
mean_pred = numpy.full(shape=len(y_test), fill_value=y_train.mean())
median_pred = numpy.full(shape=len(y_test), fill_value=y_train.median())

baseline_mean_mae = mean_absolute_error(y_test, mean_pred)
baseline_median_mae = mean_absolute_error(y_test, median_pred)

print(f"Guessing Mean Amount MAE: ${baseline_mean_mae:.2f}")
print(f"Guessing Median Amount MAE: ${baseline_median_mae:.2f}")

# Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Ridge Regression
ridge = Ridge()
ridge_scores = -cross_val_score(ridge, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
print(f"Ridge Regression Cross-Validation MAE: ${ridge_scores.mean():.2f}")

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf_scores = -cross_val_score(rf, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
print(f"Random Forest Cross-Validation MAE: ${rf_scores.mean():.2f}")
