import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from src.load_data import load_country_data
from src.preprocess import split_train_test, difference_nonstationary
from src.feature_selection import mutual_info_ranking
from src.evaluation import compute_metrics
import src.config as cfg

output_path = os.path.join(cfg.FORECASTS_DIR, "poly_lasso")
os.makedirs(output_path, exist_ok=True)

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

    for degree in cfg.POLY_DEGREES:
        for alpha in cfg.LASSO_ALPHAS:
            pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("lasso", Lasso(alpha=alpha, max_iter=5000))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            rmse, mae, mape_val = compute_metrics(y_test, y_pred)
            print(f"{country} | degree={degree}, alpha={alpha} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape_val:.2f}%")

            forecast_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}, index=X_test.index)
            forecast_df.to_csv(os.path.join(output_path, f"{country}_poly_lasso_deg{degree}_alpha{alpha}.csv"))
