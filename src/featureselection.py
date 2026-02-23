import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def correlation_filter(train_df, target_col, threshold=0.9):
    corr_matrix = train_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper_tri.columns:
        for row in upper_tri.index:
            if pd.isna(upper_tri.loc[row, col]):
                continue
            if upper_tri.loc[row, col] >= threshold:
                # drop the one less correlated with target
                target_corr = corr_matrix[target_col]
                if target_corr[row] < target_corr[col]:
                    to_drop.add(row)
                else:
                    to_drop.add(col)
    return list(to_drop)

def mutual_info_selection(train_df, target_col):
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    selected_features = mi_series[mi_series > 0.01].index.tolist()
    return selected_features, mi_series
