"""
Model Zoo and Training Module for Power System Load Type Prediction.
Contains configurations for baseline, linear, tree-based, and gradient boosting algorithms.
"""

from typing import Dict, Any
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate models ranging from trivial baselines
    to state-of-the-art gradient boosted decision trees.
    """
    models = {
        'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12,
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            random_state=random_state,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=150,
            max_depth=16,
            random_state=random_state,
            n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            random_state=random_state
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.06,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='mlogloss',
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def get_optimized_xgboost(random_state: int = 42) -> xgb.XGBClassifier:
    """
    Returns the production-tuned XGBoost model.
    """
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.1,
        eval_metric='mlogloss',
        random_state=random_state,
        n_jobs=-1
    )
