"""
Inference Module for Power System Load Type Prediction.
Provides a clean, callable interface for predicting load types on new, incoming data.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List

from src.preprocessing import clean_raw_data
from src.features import engineer_features


class LoadTypePredictor:
    """
    Production-ready inference wrapper for Power System Load Type prediction.
    """

    def __init__(self, model_path: str = 'models/best_load_type_model.joblib'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Please train the model first.")
        
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.feature_cols = artifact['feature_cols']
        self.label_mapping = artifact['label_mapping']
        self.inv_label_mapping = artifact['inv_label_mapping']
        self.metrics = artifact.get('metrics', {})

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a raw dataframe with required sensor and timestamp columns,
        executes preprocessing and feature engineering, and returns predictions.
        """
        cleaned_df = clean_raw_data(df_raw)
        feat_df, _ = engineer_features(cleaned_df)

        # Ensure all required features are present
        X = feat_df[self.feature_cols]

        pred_codes = self.model.predict(X)
        pred_labels = [self.inv_label_mapping[code] for code in pred_codes]

        result_df = df_raw.copy()
        result_df['Predicted_Load_Type'] = pred_labels

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
            for idx, label_name in sorted(self.inv_label_mapping.items()):
                result_df[f'Probability_{label_name}'] = probs[:, idx]

        return result_df


def predict_sample(sample_dict: Dict[str, Any], model_path: str = 'models/best_load_type_model.joblib') -> Dict[str, Any]:
    """
    Predicts load type for a single dictionary sample or small batch.
    """
    predictor = LoadTypePredictor(model_path)
    df_sample = pd.DataFrame([sample_dict])
    res_df = predictor.predict(df_sample)
    return res_df.iloc[0].to_dict()
