"""Utility functions for microbiome prediction."""
import numpy as np
import matplotlib.pyplot as plt


def extract_top_markers(importance, feature_names, top_n=20, direction='positive'):
    """Extract top microbial markers based on importance scores.

    Parameters:
        importance: Array of importance scores
        feature_names: List of feature names
        top_n: Number of top markers to extract
        direction: 'positive' for highest, 'negative' for lowest

    Returns:
        List of (feature_name, score) tuples
    """
    if len(importance) != len(feature_names):
        raise ValueError("importance and feature_names must have same length")

    # Sort by importance
    if direction == 'positive':
        sorted_idx = np.argsort(importance)[::-1]
    else:
        sorted_idx = np.argsort(importance)

    top_idx = sorted_idx[:top_n]
    markers = [(feature_names[i], float(importance[i])) for i in top_idx]
    return markers


def plot_feature_importance(importance, feature_names=None, top_n=20,
                           direction='positive', save_path=None, figsize=(10, 8)):
    """Plot feature importance as horizontal bar chart."""
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(len(importance))]

    markers = extract_top_markers(importance, feature_names, top_n=top_n, direction=direction)
    names = [m[0] for m in markers]
    scores = [m[1] for m in markers]

    plt.figure(figsize=figsize)
    y_pos = np.arange(len(names))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, names)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Microbial Markers')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_shap_summary(shap_values, feature_names=None, save_path=None, figsize=(10, 8)):
    """Plot SHAP summary beeswarm plot."""
    try:
        import shap
    except ImportError:
        print("shap library required for SHAP summary plot. Install with: pip install shap")
        return

    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def learning_curve(estimator, X, y, cv=5, scoring='accuracy', train_sizes=None,
                   save_path=None, figsize=(10, 6)):
    """Plot learning curve for model evaluation."""
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        n_jobs=-1, train_sizes=train_sizes
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=figsize)
    plt.plot(train_sizes_abs, train_mean, 'o-', label='Training score', color='blue')
    plt.plot(train_sizes_abs, test_mean, 'o-', label='Cross-validation score', color='orange')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring)
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None, figsize=(8, 6)):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()