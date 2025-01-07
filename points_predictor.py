from DataProcessing import DataProcessing
from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
import xgboost as xgb
from xgboost import callback
from sklearn.model_selection import RandomizedSearchCV, KFold
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

# Split data by years
train_data = df_b.data[df_b.data['Year'].isin([2021, 2022])]
#train_data.to_csv('train_data.csv', index=False)
val_data = df_b.data[df_b.data['Year'] == 2023]
test_data = df_b.data[df_b.data['Year'] == 2024]

# Prepare training data
X_train = train_data.drop(columns=["TotalPoints", "PlayerName", "Year"])
y_train = train_data["TotalPoints"]
names_train = train_data["PlayerName"]

# Prepare validation data
X_val = val_data.drop(columns=["TotalPoints", "PlayerName", "Year"])
y_val = val_data["TotalPoints"]
names_val = val_data["PlayerName"]

# Prepare test data
X_test = test_data.drop(columns=["TotalPoints", "PlayerName", "Year"])
y_test = test_data["TotalPoints"]
names_test = test_data["PlayerName"]


# 1. Define your base model (no advanced hyperparams yet):
base_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',   # often faster
    random_state=42
)

# 2. Set up a parameter distribution
param_dist = {
    'n_estimators':      [400, 500, 600, 700],
    'learning_rate':     [0.045, 0.05, 0.055, 0.06],
    'max_depth':         [1, 2, 3, 5],
    'min_child_weight':  [2, 3, 4],
    'subsample':         [0.65, 0.7, 0.75, 0.8],
    'colsample_bytree':  [0.8, 1.0],
    'gamma':             [0.05, 0.1, 0.015],
    'reg_alpha':         [0, 0.1, 1],
    'reg_lambda':        [0.4, 0.5, 0.6]
}
### CONTINUE TO TUNE HYPERPARAMETERS
# 3. Create your KFold for cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Instantiate RandomizedSearchCV
rnd_search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_dist,
    cv=kfold,  # use your KFold here
    n_iter=20, # adjust as needed
    scoring='neg_mean_squared_error', # or 'r2'
    verbose=1,
    random_state=42
)

# 5. Fit on your TRAIN data (2021–2022)
rnd_search.fit(X_train, y_train)

# 6. Best hyperparams from the random search
print("Best Params:", rnd_search.best_params_)
print("Best Score:", rnd_search.best_score_)

# 7. Evaluate best model on the validation set (2023)
best_model = rnd_search.best_estimator_
val_pred = best_model.predict(X_val)
val_rmse = root_mean_squared_error(y_val, val_pred)
val_r2 = r2_score(y_val, val_pred)
print(f"Validation RMSE on 2023: {val_rmse:.2f}")
print(f"Validation R2 on 2023: {val_r2:.2f}")
























# # 3. Define the model
# model = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     n_estimators=1000,
#     learning_rate=0.01,
#     max_depth=5,
#     min_child_weight=1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     gamma=0,
#     reg_alpha=0.1,
#     reg_lambda=1,
#     random_state=42
# )

# # Cross validation on training data only (2021-2022)
# rmse_scores, r2_scores = [], []
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Changed to KFold
# for train_idx, val_idx in kfold.split(X_train):  # Removed y_train parameter
#     X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#     y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
#     model.fit(X_fold_train, y_fold_train)
#     fold_pred = model.predict(X_fold_val)
#     rmse_scores.append(root_mean_squared_error(y_fold_val, fold_pred))
#     r2_scores.append(r2_score(y_fold_val, fold_pred))

# print(f"Cross-validation RMSE (2021-2022): {np.mean(rmse_scores):.2f}")
# print(f"Cross-validation R2 (2021-2022): {np.mean(r2_scores):.2f}")

# # Train final model on all 2021-2022 data
# model.fit(X_train, y_train)

# # Validate on 2023
# val_predictions = model.predict(X_val)
# val_rmse = root_mean_squared_error(y_val, val_predictions)
# val_r2 = r2_score(y_val, val_predictions)
# print(f"\nValidation RMSE (2023): {val_rmse:.2f}")
# print(f"Validation R2 (2023): {val_r2:.2f}")

# # Test on 2024
# final_predictions = model.predict(X_test)
# test_rmse = root_mean_squared_error(y_test, final_predictions)
# test_r2 = r2_score(y_test, final_predictions)
# print(f"\nTest RMSE (2024): {test_rmse:.2f}")
# print(f"Test R2 (2024): {test_r2:.2f}")

# # Create results DataFrame for 2024 predictions
# results_df = pd.DataFrame({
#     'PlayerName': names_test,
#     'Actual_Points': y_test,
#     'Predicted_Points': final_predictions,
#     'Difference': y_test - final_predictions,
#     'Percent_Diff': ((final_predictions - y_test) / y_test) * 100
# }).sort_values('Predicted_Points', ascending=False)

# # Display formatted results
# pd.set_option('display.max_rows', 50)
# pd.set_option('display.float_format', lambda x: '%.2f' % x)
# print("\nTop Players by Predicted Points:")
# print(results_df[['PlayerName', 'Predicted_Points', 'Actual_Points', 'Percent_Diff']].head(20))

# # Save sorted results
# results_df.to_csv('model_predictions_sorted.csv', index=False)
