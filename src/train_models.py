from src.load_data import load_features_and_target
from src.preprocess import split_country_data
from src.train_model import train_ridge, train_lasso, train_rf, train_xgb, train_svr
from config import COUNTRIES, FEATURES_DIR, TARGET_DIR

df = load_features_and_target(FEATURES_DIR, f"{TARGET_DIR}/resourceproductivity.csv")

for country in COUNTRIES:
    X_train, X_test, y_train, y_test = split_country_data(df, country)
    
    # Train each model
    ridge_model = train_ridge(X_train, y_train)
    lasso_model = train_lasso(X_train, y_train)
    rf_model = train_rf(X_train, y_train)
    xgb_model = train_xgb(X_train, y_train)
    svr_model = train_svr(X_train, y_train)

    print(f"{country} models trained")
