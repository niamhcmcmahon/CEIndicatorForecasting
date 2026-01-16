import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.feature_selection import mutual_info_regression
import numpy as np


# Load data
def load_country_data(country, indicator, features_dir, target_dir, dropna=True):
    """
    Load feature and target CSVs for a given country.
    Returns merged DataFrame with TIME_PERIOD index.
    """
    data_dict = {}
    
    # Load features
    for filename in os.listdir(features_dir):
        if filename.endswith('.csv'):
            var_name = filename.replace('.csv', '')
            df = pd.read_csv(os.path.join(features_dir, filename), usecols=['geo','TIME_PERIOD','OBS_VALUE'])
            country_df = df[df['geo'] == country].copy()
            country_df['TIME_PERIOD'] = pd.to_datetime(country_df['TIME_PERIOD'], format='%Y', errors='coerce')
            if dropna:
                country_df.dropna(subset=['TIME_PERIOD'], inplace=True)
            country_df.rename(columns={'OBS_VALUE': var_name}, inplace=True)
            data_dict[var_name] = country_df[['TIME_PERIOD', var_name]]
    
    # Load target (assume first file in target_dir)
    target_files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    if target_files:
        target_file = target_files[0]
        target_name = target_file.replace('.csv','')
        df = pd.read_csv(os.path.join(target_dir, target_file), usecols=['geo','TIME_PERIOD','OBS_VALUE'])
        country_df = df[df['geo']==country].copy()
        country_df['TIME_PERIOD'] = pd.to_datetime(country_df['TIME_PERIOD'], format='%Y', errors='coerce')
        if dropna:
            country_df.dropna(subset=['TIME_PERIOD'], inplace=True)
        country_df.rename(columns={'OBS_VALUE': target_name}, inplace=True)
        data_dict[target_name] = country_df[['TIME_PERIOD', target_name]]
    
    # Merge all
    merged_data = None
    for var, df_var in data_dict.items():
        if merged_data is None:
            merged_data = df_var
        else:
            merged_data = pd.merge(merged_data, df_var, on='TIME_PERIOD', how='outer')
    
    if merged_data is None or merged_data.empty:
        return None
    
    merged_data.set_index('TIME_PERIOD', inplace=True)
    merged_data = merged_data.sort_index()
    merged_data = merged_data.interpolate(method='linear')
    
    return merged_data

# Stationarity Assessment
def check_stationarity(series, diff=True):
    """ADF test. Optionally difference series if non-stationary."""
    result = adfuller(series.dropna())
    pval = result[1]
    stationary = pval < 0.05
    if not stationary and diff:
        series = series.diff().dropna()
        stationary = True  # after differencing
    return series, stationary

# Feature Selection
def clean_features(train_df, target_name, corr_threshold=0.9, mi_threshold=0.01):
    """
    - Remove highly correlated features (>0.9)
    - Rank features by mutual information with target
    - Return cleaned train and test feature list
    """
    corr_matrix = train_df.corr()
    features = [col for col in train_df.columns if col != target_name]
    target_corr = corr_matrix.loc[features, target_name].abs()
    feature_corr = corr_matrix.loc[features, features].abs()
    upper_tri = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper_tri.columns:
        for row in upper_tri.index:
            val = upper_tri.loc[row,col]
            if pd.isna(val): continue
            if val >= corr_threshold:
                if target_corr[row] < target_corr[col]:
                    to_drop.add(row)
                else:
                    to_drop.add(col)
    
    cleaned_train = train_df.drop(columns=list(to_drop))
    
    # MI Ranking
    X_train = cleaned_train.drop(columns=[target_name])
    y_train = cleaned_train[target_name]
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    
    selected_features = mi_series[mi_series > mi_threshold].index.tolist()
    
    return cleaned_train[selected_features + [target_name]], selected_features, mi_series

# Functions to visualise ACF plots and Correlation Matrices
def plot_acf_pacf(series, lags=6, save_path=None, title='ACF/PACF'):
    """Plot ACF and PACF"""
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    plot_acf(series, ax=axes[0], lags=lags)
    plot_pacf(series, ax=axes[1], lags=lags, method="ywm")
    axes[0].set_title(f'ACF: {title}')
    axes[1].set_title(f'PACF: {title}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_corr_matrix(df, save_path=None, title='Correlation Matrix'):
    """Plot correlation heatmap"""
    corr = df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={'label':'Correlation Coefficient'}, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
