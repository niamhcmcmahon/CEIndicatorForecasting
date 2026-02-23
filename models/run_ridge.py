import pandas as pd
from src.config import RAW_DATA_DIR
from src.rolling_origin import rolling_origin_forecast
from src.metrics import rmse, mae, mape
from src.bootstrap import gaussian_bootstrap


#Load data
df = pd.read_csv(RAW_DATA_DIR / "resourceproductivity.csv")

# Adjust if needed
target_col = "ResourceProductivity"
feature_cols = [col for col in df.columns if col != target_col]

#train/test 80/20
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

#scaling 
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# model
model = Ridge(alpha=1.0)

predictions, true_values, residuals = rolling_origin_forecast(
    model,
    train_df,
    test_df,
    feature_cols,
    target_col
)

#eval
print("RMSE:", rmse(true_values, predictions))
print("MAE:", mae(true_values, predictions))
print("MAPE:", mape(true_values, predictions))

#CI
lower, upper = gaussian_bootstrap(predictions, residuals)

print("Bootstrap CI computed.")
