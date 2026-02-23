import pandas as pd
import os

def load_country_data(indicator: str, countries: list):
    base_dir = 'data/raw/indicator/resourceproductivity'
    features_dir = os.path.join(base_dir, 'features')
    target_dir = os.path.join(base_dir, 'target')

    country_data_dict = {}

    for country in countries:
        data_dict = {}
        # Load features
        for f in os.listdir(features_dir):
            if f.endswith('.csv'):
                var_name = f.replace('.csv', '')
                df = pd.read_csv(os.path.join(features_dir, f), usecols=['geo', 'TIME_PERIOD', 'OBS_VALUE'])
                df = df[df['geo'] == country]
                df.rename(columns={'OBS_VALUE': var_name}, inplace=True)
                df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y', errors='coerce')
                df.dropna(subset=['TIME_PERIOD'], inplace=True)
                data_dict[var_name] = df[['TIME_PERIOD', var_name]]

        # Load target
        for f in os.listdir(target_dir):
            if f.endswith('.csv'):
                var_name = f.replace('.csv', '')
                df = pd.read_csv(os.path.join(target_dir, f), usecols=['geo', 'TIME_PERIOD', 'OBS_VALUE'])
                df = df[df['geo'] == country]
                df.rename(columns={'OBS_VALUE': var_name}, inplace=True)
                df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y', errors='coerce')
                df.dropna(subset=['TIME_PERIOD'], inplace=True)
                data_dict[var_name] = df[['TIME_PERIOD', var_name]]

        # Merge all variables
        merged = None
        for df in data_dict.values():
            merged = df if merged is None else pd.merge(merged, df, on='TIME_PERIOD', how='outer')

        if merged is not None:
            merged.sort_values('TIME_PERIOD', inplace=True)
            merged.set_index('TIME_PERIOD', inplace=True)
            merged = merged.interpolate(method='linear')  # fill missing
            country_data_dict[country] = merged

    return country_data_dict
