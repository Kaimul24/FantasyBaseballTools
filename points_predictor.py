from DataProcessing import DataProcessing
from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
import xgboost as xgb
from xgboost import callback
from sklearn.model_selection import train_test_split, KFold  # Change import
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np

# 1. Load or merge your advanced metrics + historical fantasy data
df_b = DataProcessing(PositionCategory.BATTER)
df_b.filter_data()
df_b.calc_fantasy_points()
df_b.data = df_b.data.drop(columns=['1B', '2B', '3B', 'HR', 'R', 'RBI', 'SB', 'HBP'])

player_names = df_b.data['PlayerName'].copy()
X = df_b.data.drop(columns=["TotalPoints", "PlayerName"])
y = df_b.data["TotalPoints"]

# 3. Define the model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

# Split the data once for final evaluation
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, player_names, test_size=0.2, random_state=42
)

# Cross validation for model evaluation
rmse_scores, r2_scores = [], []
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Changed to KFold
for train_idx, val_idx in kfold.split(X_train):  # Removed y_train parameter
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    fold_pred = model.predict(X_fold_val)
    rmse_scores.append(root_mean_squared_error(y_fold_val, fold_pred))
    r2_scores.append(r2_score(y_fold_val, fold_pred))

print(f"Average RMSE across folds: {np.mean(rmse_scores):.2f}")
print(f"Average R2 across folds: {np.mean(r2_scores):.2f}")

# Final model training and prediction
model.fit(X_train, y_train)
final_predictions = model.predict(X_test)

# Create results DataFrame
results_df = pd.DataFrame({
    'PlayerName': names_test,
    'Actual_Points': y_test,
    'Predicted_Points': final_predictions,
    'Difference': y_test - final_predictions,
    'Percent_Diff': ((final_predictions - y_test) / y_test) * 100
}).sort_values('Predicted_Points', ascending=False)

# Display formatted results
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print("\nTop Players by Predicted Points:")
print(results_df[['PlayerName', 'Predicted_Points', 'Actual_Points', 'Percent_Diff']].head(20))

# Save sorted results
results_df.to_csv('model_predictions_sorted.csv', index=False)
