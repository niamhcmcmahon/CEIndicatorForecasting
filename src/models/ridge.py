from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
import pandas as pd


def rolling_origin_poly_ridge(
    train_df, 
    test_df, 
    feature_cols, 
    target_col, 
    poly_degrees=[1,2], 
    alphas=[0.5,1,10], 
    n_boot=1000
):
    """
    Perform rolling-origin forecast using Polynomial Ridge regression with bootstrap uncertainty.
    Returns the best forecast DataFrame and metrics.
    """
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    best_test_rmse = np.inf
    best_forecast_df = None
    best_params = None
    best_train_fitted = None
    best_train_true = None

    from itertools import product
    for degree, alpha in product(poly_degrees, alphas):
        # Prepare polynomial transformer
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Initial training residuals for first bootstrap
        poly_train_full = poly.fit_transform(train_df_scaled[feature_cols].values)
        y_train_full = train_df_scaled[target_col].values
        model_full = Ridge(alpha=alpha, max_iter=10000, random_state=42)
        model_full.fit(poly_train_full, y_train_full)
        train_residuals = y_train_full - model_full.predict(poly_train_full)
        train_resid_std = np.std(train_residuals)

        # Rolling-origin
        train_window = train_df_scaled.copy()
        predictions, true_values, residuals = [], [], []
        train_fitted_preds, train_true = [], []

        for i in range(len(test_df_scaled)):
            X_train = poly.fit_transform(train_window[feature_cols].values)
            y_train = train_window[target_col].values

            X_test = poly.transform(test_df_scaled[feature_cols].iloc[i:i+1].values)
            y_test = test_df_scaled[target_col].iloc[i]

            model = Ridge(alpha=alpha, max_iter=10000, random_state=42)
            model.fit(X_train, y_train)

            fitted_train = model.predict(X_train)
            train_fitted_preds.extend(fitted_train)
            train_true.extend(y_train)

            y_pred = model.predict(X_test)[0]
            predictions.append(y_pred)
            true_values.append(y_test)

            if i == 0:
                resid_std = train_resid_std
            else:
                resid_std = np.std(residuals)

            y_boot = y_pred + np.random.normal(0, resid_std, size=n_boot)
            if i == 0:
                bootstrap_array = y_boot[None,:]
            else:
                bootstrap_array = np.vstack([bootstrap_array, y_boot])

            residuals.append(y_test - y_pred)
            train_window = pd.concat([train_window, test_df_scaled.iloc[i:i+1]])

        test_rmse = np.sqrt(mean_squared_error(true_values, predictions))
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_params = {'poly_degree': degree, 'alpha': alpha}
            best_forecast_df = pd.DataFrame({
                'True': true_values,
                'Predicted': predictions,
                'Mean_Bootstrap': bootstrap_array.mean(axis=1),
                'Lower95_CI': np.percentile(bootstrap_array, 2.5, axis=1),
                'Upper95_CI': np.percentile(bootstrap_array, 97.5, axis=1)
            }, index=test_df_scaled.index)
            best_train_fitted = np.array(train_fitted_preds)
            best_train_true = np.array(train_true)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(best_train_true, best_train_fitted))
    train_mae = mean_absolute_error(best_train_true, best_train_fitted)
    train_mape = mape(best_train_true, best_train_fitted)
    test_mae = mean_absolute_error(best_forecast_df['True'], best_forecast_df['Predicted'])
    test_mape = mape(best_forecast_df['True'], best_forecast_df['Predicted'])

    metrics = {
        'best_params': best_params,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_mape': train_mape,
        'test_rmse': best_test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape
    }

    return best_forecast_df, metrics
