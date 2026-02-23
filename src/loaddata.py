import os
import pandas as pd
from src.config import COUNTRIES, FEATURES_DIR, TARGET_DIR

def load_data(feature_files, target_file):
    """
    Load feature and target CSVs for countries specified in config.
    Returns a dictionary {country: merged_dataframe}.
    """
    all_data = {}

    for country in COUNTRIES:
        data_dict = {}

        # Load features
        for filename in feature_files:
            var_name = filename.replace('.csv', '')
            filepath = os.path.join(FEATURES_DIR, filename)
            df = pd.read_csv(filepath, usecols=['geo', 'TIME_PERIOD', 'OBS_VALUE'])
            country_data = df[df['geo'] == country].copy()
            country_data.rename(columns={'OBS_VALUE': var_name}, inplace=True)
            data_dict[var_name] = country_data[['TIME_PERIOD', var_name]]

        # Load target
        target_name = target_file.replace('.csv', '')
        target_filepath = os.path.join(TARGET_DIR, target_file)
        target_df = pd.read_csv(target_filepath, usecols=['geo', 'TIME_PERIOD', 'OBS_VALUE'])
        target_data = target_df[target_df['geo'] == country].copy()
        target_data.rename(columns={'OBS_VALUE': target_name}, inplace=True)
        data_dict[target_name] = target_data[['TIME_PERIOD', target_name]]

        # Merge all dataframes
        merged_data = data_dict[list(data_dict.keys())[0]]
        for var in list(data_dict.keys())[1:]:
            merged_data = merged_data.merge(data_dict[var], on='TIME_PERIOD', how='outer')

        # Process datetime, filter, interpolate, drop NaNs
        merged_data['TIME_PERIOD'] = pd.to_datetime(merged_data['TIME_PERIOD'], format='%Y')
        merged_data.set_index('TIME_PERIOD', inplace=True)
        merged_data = merged_data[merged_data.index.year <= 2018]
        merged_data = merged_data.interpolate(method='linear').dropna()

        all_data[country] = merged_data

    return all_data
