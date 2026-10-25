"""
Feature Engineering Module for Power System Load Type Prediction.
Transforms cleaned time-series data into domain-informed electrical,
temporal, cyclical, and causal rolling/lag features.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs comprehensive features:
    1. Temporal & Calendar Features.
    2. Seasonal & Tariff Period Indicators.
    3. Cyclical Trigonometric Encodings (Hour, DayOfWeek, Month).
    4. Electrical Domain Power Relations (Apparent Power S, Q_total, Q_diff, Ratios).
    5. Causal Lag & Rolling Statistics (zero-lookahead, backward only).
    """
    feat_df = df.copy()

    # Base active and reactive powers (using cleaned series)
    p = feat_df['Usage_kWh_clean'] if 'Usage_kWh_clean' in feat_df.columns else feat_df['Usage_kWh']
    q_lag = feat_df['Lagging_Current_Reactive.Power_kVarh_clean'] if 'Lagging_Current_Reactive.Power_kVarh_clean' in feat_df.columns else feat_df['Lagging_Current_Reactive.Power_kVarh']
    q_lead = feat_df['Leading_Current_Reactive_Power_kVarh_clean'] if 'Leading_Current_Reactive_Power_kVarh_clean' in feat_df.columns else feat_df['Leading_Current_Reactive_Power_kVarh']

    # 1. Electrical Domain Features
    feat_df['Q_total'] = q_lag + q_lead
    feat_df['Q_diff'] = q_lag - q_lead
    feat_df['Apparent_Power_kVA'] = np.sqrt(p**2 + (q_lag + q_lead)**2)
    feat_df['Active_Reactive_Ratio'] = p / (feat_df['Q_total'] + 1e-4)
    feat_df['Calculated_Power_Factor'] = (p / (np.sqrt(p**2 + q_lag**2) + 1e-6)) * 100.0
    feat_df['Calculated_Power_Factor'] = feat_df['Calculated_Power_Factor'].clip(0.0, 100.0)

    # 2. Seasonal & Tariff Indicators
    # Winter tariff period: November, December, January, February
    feat_df['Is_Winter'] = feat_df['Month'].isin([11, 12, 1, 2]).astype(int)
    # Season category: 0=Winter, 1=Spring, 2=Summer, 3=Autumn
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 0}
    feat_df['Season'] = feat_df['Month'].map(season_map)

    # 3. Cyclical Transformations
    feat_df['sin_hour'] = np.sin(2 * np.pi * feat_df['Hour'] / 24.0)
    feat_df['cos_hour'] = np.cos(2 * np.pi * feat_df['Hour'] / 24.0)
    feat_df['sin_dayofweek'] = np.sin(2 * np.pi * feat_df['DayOfWeek'] / 7.0)
    feat_df['cos_dayofweek'] = np.cos(2 * np.pi * feat_df['DayOfWeek'] / 7.0)
    feat_df['sin_month'] = np.sin(2 * np.pi * feat_df['Month'] / 12.0)
    feat_df['cos_month'] = np.cos(2 * np.pi * feat_df['Month'] / 12.0)

    # 4. Causal Temporal Dynamics (strictly backward-looking to eliminate leakage)
    feat_df['Usage_lag_1'] = p.shift(1).bfill()
    feat_df['Usage_lag_4'] = p.shift(4).bfill()      # 1 hour ago
    feat_df['Usage_lag_96'] = p.shift(96).bfill()    # 24 hours ago (same time yesterday)
    feat_df['Usage_diff_1'] = (p - feat_df['Usage_lag_1']).fillna(0)

    # Rolling window aggregations
    feat_df['Usage_rolling_mean_4'] = p.rolling(window=4, min_periods=1).mean()    # 1 hour
    feat_df['Usage_rolling_mean_24'] = p.rolling(window=24, min_periods=1).mean()  # 6 hours
    feat_df['Usage_rolling_mean_96'] = p.rolling(window=96, min_periods=1).mean()  # 24 hours
    feat_df['Usage_rolling_std_24'] = p.rolling(window=24, min_periods=1).std().fillna(0)

    feature_cols = [
        # Sensor raw & cleaned
        'Usage_kWh_clean',
        'Lagging_Current_Reactive.Power_kVarh_clean',
        'Leading_Current_Reactive_Power_kVarh_clean',
        'CO2(tCO2)_clean',
        'Lagging_Current_Power_Factor_clipped_clean',
        'Leading_Current_Power_Factor_clipped_clean',
        # Electrical Domain
        'Q_total',
        'Q_diff',
        'Apparent_Power_kVA',
        'Active_Reactive_Ratio',
        'Calculated_Power_Factor',
        # Calendar & Exact Time
        'NSM_exact',
        'Hour',
        'Minute',
        'DayOfWeek',
        'IsWeekend',
        'Month',
        'Is_Winter',
        'Season',
        # Cyclical Encodings
        'sin_hour',
        'cos_hour',
        'sin_dayofweek',
        'cos_dayofweek',
        'sin_month',
        'cos_month',
        # Causal Lags & Rolling
        'Usage_lag_1',
        'Usage_lag_4',
        'Usage_lag_96',
        'Usage_diff_1',
        'Usage_rolling_mean_4',
        'Usage_rolling_mean_24',
        'Usage_rolling_mean_96',
        'Usage_rolling_std_24'
    ]

    return feat_df, feature_cols
