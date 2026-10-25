"""
Data Preprocessing Module for Power System Load Type Prediction.
Handles data loading, cleaning, anomaly correction, missing value imputation,
and chronological train-test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

LABEL_MAPPING: Dict[str, int] = {
    'Light_Load': 0,
    'Medium_Load': 1,
    'Maximum_Load': 2
}

INV_LABEL_MAPPING: Dict[int, str] = {v: k for k, v in LABEL_MAPPING.items()}


def clean_raw_data(filepath_or_df) -> pd.DataFrame:
    """
    Cleans raw load data by:
    1. Parsing datetime objects.
    2. Removing trailing corrupted duplicate row if present.
    3. Correcting synthetic / corrupt NSM values using exact time calculation.
    4. Correcting power factors bounded within the physical [0, 100]% range.
    5. Imputing missing continuous sensor values using time-aware interpolation.
    """
    if isinstance(filepath_or_df, str):
        df = pd.read_csv(filepath_or_df)
    else:
        df = filepath_or_df.copy()

    # 1. Check and remove trailing duplicate row (index 35040 if 35041 rows)
    if len(df) == 35041 and df.iloc[-1]['Date_Time'] == df.iloc[-18]['Date_Time']:
        df = df.iloc[:35040].copy().reset_index(drop=True)

    # 2. Parse Date_Time
    df['dt'] = pd.to_datetime(df['Date_Time'], format='%d-%m-%Y %H:%M')

    # 3. Extract exact temporal components
    df['Hour'] = df['dt'].dt.hour
    df['Minute'] = df['dt'].dt.minute
    df['Day'] = df['dt'].dt.day
    df['Month'] = df['dt'].dt.month
    df['DayOfWeek'] = df['dt'].dt.dayofweek
    df['DayName'] = df['dt'].dt.day_name()
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # 4. Exact NSM (Number of Seconds from Midnight)
    # The raw dataset had simulated float noise (>86400s). Physical bounds are 0 to 85500s.
    df['NSM_exact'] = df['Hour'] * 3600 + df['Minute'] * 60

    # 5. Fix physical anomalies in Power Factors (power factor cannot exceed 100%)
    if 'Lagging_Current_Power_Factor' in df.columns:
        df['Lagging_Current_Power_Factor_clipped'] = df['Lagging_Current_Power_Factor'].clip(0.0, 100.0)
    if 'Leading_Current_Power_Factor' in df.columns:
        df['Leading_Current_Power_Factor_clipped'] = df['Leading_Current_Power_Factor'].clip(0.0, 100.0)

    # 6. Impute continuous sensor features using time-series linear interpolation
    sensor_cols = [
        'Usage_kWh',
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor_clipped',
        'Leading_Current_Power_Factor_clipped'
    ]
    for col in sensor_cols:
        if col in df.columns:
            # Interpolate smoothly along 15-min intervals, then backfill/forward-fill edges
            df[f'{col}_clean'] = df[col].interpolate(method='linear', limit_direction='both').bfill().ffill()

    # 7. Map target label if Load_Type exists
    if 'Load_Type' in df.columns:
        df['target'] = df['Load_Type'].map(LABEL_MAPPING)

    return df


def split_train_test_by_month(df: pd.DataFrame, test_month: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into chronological train and test sets.
    Months < test_month form the training set (Months 1-11).
    Month == test_month forms the out-of-sample test set (Month 12, December).
    """
    train_df = df[df['Month'] < test_month].copy().reset_index(drop=True)
    test_df = df[df['Month'] == test_month].copy().reset_index(drop=True)
    return train_df, test_df
