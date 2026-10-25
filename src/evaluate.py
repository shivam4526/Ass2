"""
Evaluation and Diagnostic Module for Power System Load Type Prediction.
Computes multi-class classification metrics and creates publication-grade visual artifacts.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# Professional plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes all classification evaluation metrics:
    Accuracy, Macro/Weighted Precision, Recall, F1, Cohen Kappa, MCC, ROC-AUC, Log-Loss.
    """
    metrics = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Macro Precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'Macro Recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'Macro F1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'Weighted Precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'Weighted Recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'Weighted F1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'Cohen Kappa': float(cohen_kappa_score(y_true, y_pred)),
        'Matthews Corrcoef': float(matthews_corrcoef(y_true, y_pred))
    }

    if y_prob is not None:
        try:
            metrics['ROC-AUC (OVR)'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            metrics['Log-Loss'] = float(log_loss(y_true, y_prob))
        except Exception:
            pass

    return metrics


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots dual confusion matrices: raw counts and row-normalized percentages.
    """
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Raw counts
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=True
    )
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Predicted Label', fontsize=11, labelpad=8)
    axes[0].set_ylabel('True Label', fontsize=11, labelpad=8)

    # Normalized percentages
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar=True
    )
    axes[1].set_title('Confusion Matrix (Normalized Recall)', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('Predicted Label', fontsize=11, labelpad=8)
    axes[1].set_ylabel('True Label', fontsize=11, labelpad=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots One-vs-Rest ROC curves with AUC for each class and macro-average.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=2.2,
            label=f'{class_names[i]} (AUC = {roc_auc:.4f})'
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance Baseline (AUC = 0.5000)')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, labelpad=8)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, labelpad=8)
    ax.set_title('Multi-Class One-vs-Rest ROC Curves', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower right', frameon=True, fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_feature_importances(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots top N feature importances from a tree-based model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier'), 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
    else:
        raise ValueError("Model does not provide feature_importances_")

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax, palette='mako_r')
    ax.set_title(f'Top {top_n} Most Influential Features (Gain/Importance)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Relative Importance Score', fontsize=11, labelpad=8)
    ax.set_ylabel('Feature Name', fontsize=11, labelpad=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
