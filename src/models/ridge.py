import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from src.load_data import load_country_data
from src.preprocess import split_train_test, difference_nonstationary
from src.feature_selection import mutual_info_ranking
from src.evaluation import compute_metrics
import src.config as cfg

# Settings from config.py
COUNTRIES = cfg.COUNTRIES
TARGET_COL = 'resourceproductivity'  
DEGREE = cfg.POLY_DEGREES[1] 
ALPHA = cfg.RIDGE_ALPHAS[1]   
OUTPUT_PATH = f"{cfg.FORECASTS_DIR}/poly_ridge/"

for country in COUNTRIES:
    print(f'Processing {country}')
    data = load_country_data(TARGET_COL, [country])[country]

    # Preprocess
    train, test = split_train_test(data, split_ratio=cfg.TRAIN_TEST_SPLIT_RATIO)
    diff_train = difference_nonstationary(train)
    diff_test = test[diff_train.columns]

    X_train = diff_train.drop(columns=[TARGET_COL])
    y_train = diff_train[TARGET_COL]
    X_test = diff_test.drop(columns=[TARGET_COL])
    y_test = diff_test[TARGET_COL]

    # Feature selection
    mi_scores = mutual_info_ranking(X_train, y_train, threshold=cfg.MUTUAL_INFO_THRESHOLD)
    top_features = mi_scores.head(10).index.tolist()
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # Pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=ALPHA))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    rmse, mae, mape_val = compute_metrics(y_test, y_pred)
    print(f'{country} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape_val:.2f}%')

    # Save forecasts
    forecast_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}, index=X_test.index)
    forecast_df.to_csv(f'{OUTPUT_PATH}{country}_poly_ridge.csv', index=True)
