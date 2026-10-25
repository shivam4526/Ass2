"""
End-to-End Pipeline Execution Script.
Executes data preprocessing, feature engineering, cross-validation,
model comparison, final model evaluation on test month, and model artifact saving.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

from src.preprocessing import clean_raw_data, split_train_test_by_month, LABEL_MAPPING, INV_LABEL_MAPPING
from src.features import engineer_features
from src.models import get_candidate_models, get_optimized_xgboost
from src.evaluate import (
    evaluate_predictions,
    plot_confusion_matrices,
    plot_multiclass_roc_curves,
    plot_feature_importances
)


def run_pipeline(data_path: str = 'data/load_data.csv'):
    print("=" * 70)
    print("STARTING END-TO-END LOAD TYPE PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Preprocessing
    print(f"\n[1/7] Loading and cleaning dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    print(f"  Raw dataset shape: {raw_df.shape}")
    cleaned_df = clean_raw_data(raw_df)
    print(f"  Cleaned dataset shape: {cleaned_df.shape}")

    # 2. Feature Engineering
    print("\n[2/7] Engineering domain, cyclical, and causal rolling features...")
    feat_df, feature_cols = engineer_features(cleaned_df)
    print(f"  Total engineered features: {len(feature_cols)}")

    # 3. Chronological Train-Test Split (Per problem requirements)
    print("\n[3/7] Splitting dataset: Months 1-11 (Train) | Month 12 (Test)...")
    train_df, test_df = split_train_test_by_month(feat_df, test_month=12)
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_test, y_test = test_df[feature_cols], test_df['target']
    print(f"  Training set records: {X_train.shape[0]} ({train_df['dt'].min()} to {train_df['dt'].max()})")
    print(f"  Test set records:     {X_test.shape[0]} ({test_df['dt'].min()} to {test_df['dt'].max()})")

    # 4. Cross-Validation on Training Data
    print("\n[4/7] Performing 5-Fold Stratified Cross-Validation on Training Data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_models = get_candidate_models(random_state=42)

    cv_results = {}
    for name, model in candidate_models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_results[name] = (scores.mean(), scores.std())
        print(f"  {name:25s} | 5-Fold CV Macro F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 5. Out-of-Sample Test Evaluation
    print("\n[5/7] Evaluating candidate models on the Unseen Test Month (December 2018)...")
    benchmark_records = []
    class_names = ['Light_Load', 'Medium_Load', 'Maximum_Load']

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        metrics = evaluate_predictions(y_test, y_pred, y_prob, class_names)
        metrics['Model'] = name
        metrics['CV Macro F1'] = cv_results[name][0]
        benchmark_records.append(metrics)

    benchmark_df = pd.DataFrame(benchmark_records)
    cols_order = ['Model', 'CV Macro F1', 'Accuracy', 'Macro F1', 'Weighted F1', 'ROC-AUC (OVR)', 'Cohen Kappa']
    summary_table = benchmark_df[[c for c in cols_order if c in benchmark_df.columns]]
    print("\n" + "=" * 70)
    print("MODEL BENCHMARK COMPARISON ON TEST SET (DECEMBER 2018)")
    print("=" * 70)
    print(summary_table.to_string(index=False))

    # Save benchmark table
    os.makedirs('reports', exist_ok=True)
    summary_table.to_csv('reports/model_benchmark_comparison.csv', index=False)
    print("  Saved benchmark summary to 'reports/model_benchmark_comparison.csv'")

    # 6. Fine-Tuned Best Model (XGBoost)
    print("\n[6/7] Training Production-Grade Optimized XGBoost Model...")
    best_model = get_optimized_xgboost(random_state=42)
    best_model.fit(X_train, y_train)

    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)

    final_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob, class_names)
    print("\nFinal Test Metrics (Optimized XGBoost):")
    for k, v in final_metrics.items():
        print(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names, digits=4))

    # Generate and save diagnostic figures
    print("\nGenerating diagnostic plots...")
    plot_confusion_matrices(y_test, y_test_pred, class_names, save_path='reports/confusion_matrix.png')
    plot_multiclass_roc_curves(y_test, y_test_prob, class_names, save_path='reports/roc_curves.png')
    plot_feature_importances(best_model, feature_cols, top_n=15, save_path='reports/feature_importance.png')
    print("  Saved plots to 'reports/' directory.")

    # 7. Model Serialization
    print("\n[7/7] Serializing Production Model Artifact...")
    os.makedirs('models', exist_ok=True)
    model_artifact = {
        'model': best_model,
        'feature_cols': feature_cols,
        'label_mapping': LABEL_MAPPING,
        'inv_label_mapping': INV_LABEL_MAPPING,
        'metrics': final_metrics
    }
    model_save_path = 'models/best_load_type_model.joblib'
    joblib.dump(model_artifact, model_save_path)
    print(f"  Model artifact saved to '{model_save_path}'")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return benchmark_df, final_metrics


if __name__ == '__main__':
    run_pipeline('data/load_data.csv')
