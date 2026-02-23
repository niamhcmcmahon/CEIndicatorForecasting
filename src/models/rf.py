import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.load_data import load_country_data
from src.preprocess import split_train_test, difference_nonstationary
from src.feature_selection import mutual_info_ranking
from src.evaluation import compute_metrics
import src.config as cfg
from itertools import product

output_path = os.path.join(cfg.FORECASTS_DIR, "rf")
os.makedirs(output_path, exist_ok=True)

param_grid = list(product(
    cfg.RF_PARAMS['n_estimators'],
    cfg.RF_PARAMS['max_depth'],
    cfg.RF_PARAMS['min_samples_split'],
    cfg.RF_PARAMS['min_samples_leaf']
))

for country in cfg.COUNTRIES:
    print(f"Processing {country}")
    data = load_country_data("resourceproductivity", [country])[country]

    train, test = split_train_test(data, split_ratio=cfg.TRAIN_TEST_SPLIT_RATIO)
    diff_train = difference_nonstationary(train)
    diff_test = test[diff_train.columns]

    X_train = diff_train.drop(columns=["resourceproductivity"])
    y_train = diff_train["resourceproductivity"]
    X_test = diff_test.drop(columns=["resourceproductivity"])
    y_test = diff_test["resourceproductivity"]

    mi_scores = mutual_info_ranking(X_train, y_train, threshold=cfg.MUTUAL_INFO_THRESHOLD)
    top_features = mi_scores.head(10).index.tolist()
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    for n_estimators, max_depth, min_split, min_leaf in param_grid:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse, mae, mape_val = compute_metrics(y_test, y_pred)
        print(f"{country} | n={n_estimators}, depth={max_depth} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape_val:.2f}%")

        forecast_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}, index=X_test.index)
        forecast_df.to_csv(os.path.join(output_path, f"{country}_rf_n{n_estimators}_depth{max_depth}.csv"))
