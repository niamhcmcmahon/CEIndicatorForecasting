import pandas as pd
import os

countries = ['Belgium', 'Bulgaria', 'Czechia', 'Denmark', 'Germany', 'Estonia', 'Ireland',
             'Greece', 'Spain', 'France', 'Italy', 'Cyprus', 'Latvia', 'Lithuania',
             'Luxembourg', 'Hungary', 'Netherlands', 'Austria', 'Poland', 'Portugal',
             'Romania', 'Slovenia', 'Slovakia', 'Finland', 'Sweden']

def load_country_data(features_dir, target_dir, country):
    data_dict = {}
    # Load features
    for filename in os.listdir(features_dir):
        if filename.endswith('.csv'):
            var = filename.replace('.csv', '')
            df = pd.read_csv(os.path.join(features_dir, filename), usecols=['geo', 'TIME_PERIOD', 'OBS_VALUE'])
            country_df = df[df['geo'] == country].copy()
            country_df['TIME_PERIOD'] = pd.to_datetime(country_df['TIME_PERIOD'], format='%Y', errors='coerce')
            country_df.dropna(subset=['TIME_PERIOD'], inplace=True)
            country_df.rename(columns={'OBS_VALUE': var}, inplace=True)
            data_dict[var] = country_df[['TIME_PERIOD', var]]
    # Load target
    for filename in os.listdir(target_dir):
        if filename.endswith('.csv'):
            var = filename.replace('.csv', '')
            df = pd.read_csv(os.path.join(target_dir, filename), usecols=['geo', 'TIME_PERIOD', 'OBS_VALUE'])
            country_df = df[df['geo'] == country].copy()
            country_df['TIME_PERIOD'] = pd.to_datetime(country_df['TIME_PERIOD'], format='%Y', errors='coerce')
            country_df.dropna(subset=['TIME_PERIOD'], inplace=True)
            country_df.rename(columns={'OBS_VALUE': var}, inplace=True)
            data_dict[var] = country_df[['TIME_PERIOD', var]]
    return data_dict
