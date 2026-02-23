import pandas as pd


def rolling_origin_forecast(
    model,
    train_df,
    test_df,
    feature_cols,
    target_col
):
    train_window = train_df.copy()

    predictions = []
    true_values = []
    residuals = []

    for i in range(len(test_df)):
        X_train = train_window[feature_cols]
        y_train = train_window[target_col]

        X_test = test_df[feature_cols].iloc[i:i+1]
        y_test = test_df[target_col].iloc[i]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]

        predictions.append(y_pred)
        true_values.append(y_test)
        residuals.append(y_test - y_pred)

        # expand window
        train_window = pd.concat([train_window, test_df.iloc[i:i+1]])

    return predictions, true_values, residuals
