"""Fusion methods for multi-omics integration: CCA, Procrustes."""
from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy.spatial import procrustes


def _canonical_correlation(X: np.ndarray, Y: np.ndarray, n_comp: int):
    """Compute canonical correlations using eigendecomposition.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_x)
        First data matrix (centered).
    Y : ndarray of shape (n_samples, n_features_y)
        Second data matrix (centered).
    n_comp : int
        Number of canonical components to compute.

    Returns
    -------
    corrs : ndarray of shape (n_comp,)
        Canonical correlations.
    loadings_x : ndarray of shape (n_features_x, n_comp)
        Loadings for X.
    loadings_y : ndarray of shape (n_features_y, n_comp)
        Loadings for Y.
    """
    n, p, q = X.shape[0], X.shape[1], Y.shape[1]
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    S_xx = X_c.T @ X_c / (n - 1)
    S_yy = Y_c.T @ Y_c / (n - 1)
    S_xy = X_c.T @ Y_c / (n - 1)

    eps = 1e-8
    S_xx = S_xx + eps * np.eye(p)
    S_yy = S_yy + eps * np.eye(q)

    Ux, sx, Vx = np.linalg.svd(S_xx, full_matrices=False)
    Uy, sy, Vy = np.linalg.svd(S_yy, full_matrices=False)

    S_xx_inv_sqrt = Ux @ np.diag(1.0 / np.sqrt(sx + 1e-10)) @ Vx
    S_yy_inv_sqrt = Uy @ np.diag(1.0 / np.sqrt(sy + 1e-10)) @ Vy

    M = S_xx_inv_sqrt @ S_xy @ S_yy_inv_sqrt
    U, s, V = np.linalg.svd(M, full_matrices=False)

    corrs = s[:n_comp]
    loadings_x = S_xx_inv_sqrt @ U[:, :n_comp]
    loadings_y = S_yy_inv_sqrt @ V.T[:, :n_comp]

    return corrs, loadings_x, loadings_y


class CCAAnalyzer:
    """Canonical Correlation Analysis for multi-omics correlation."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self._scores: Optional[Dict[str, np.ndarray]] = None
        self._loadings: Optional[pd.DataFrame] = None
        self._correlations: Optional[np.ndarray] = None

    def fit(self, data_dict: Dict[str, pd.DataFrame]) -> 'CCAAnalyzer':
        """Fit CCA on two omics blocks."""
        if len(data_dict) != 2:
            raise ValueError("CCA requires exactly two omics blocks")
        keys = list(data_dict.keys())
        X, Y = data_dict[keys[0]].values, data_dict[keys[1]].values

        n_comp = min(self.n_components, X.shape[1], Y.shape[1])

        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)

        try:
            corrs, loadings_x, loadings_y = _canonical_correlation(X_centered, Y_centered, n_comp)
        except Exception:
            self._scores = {}
            self._loadings = pd.DataFrame()
            self._correlations = np.array([])
            return self

        self._scores = {
            keys[0]: X_centered @ loadings_x,
            keys[1]: Y_centered @ loadings_y,
        }

        # loadings_x: (n_features_X, n_comp), loadings_y: (n_features_Y, n_comp)
        # Build DataFrame: rows are CCA components, columns are features from each block
        loadings_df = pd.DataFrame(
            np.hstack([loadings_x.T, loadings_y.T]),  # (n_comp, n_features_X + n_features_Y)
            index=[f'cca{i+1}' for i in range(n_comp)],
            columns=list(data_dict[keys[0]].columns) + list(data_dict[keys[1]].columns)
        )
        loadings_df.columns = pd.MultiIndex.from_tuples([
            (keys[0], col) for col in data_dict[keys[0]].columns
        ] + [
            (keys[1], col) for col in data_dict[keys[1]].columns
        ], names=['block', 'feature'])
        self._loadings = loadings_df
        self._correlations = corrs
        return self

    def fit_transform(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Fit and return results."""
        self.fit(data_dict)
        return self.get_results()

    def transform(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Apply CCA to new data."""
        return self._scores or {}

    def get_loadings(self) -> pd.DataFrame:
        """Return canonical loadings DataFrame."""
        return self._loadings if self._loadings is not None else pd.DataFrame()

    def get_results(self) -> Dict:
        """Return full results dict."""
        return {
            'method': 'cca',
            'loadings': self._loadings if self._loadings is not None else pd.DataFrame(),
            'correlations': self._correlations if self._correlations is not None else np.array([]),
            'scores': self._scores if self._scores else {},
        }


class ProcrustesAnalyzer:
    """Procrustes analysis for PCoA coordinate alignment."""

    def __init__(self):
        self._result: Optional[Dict] = None

    def fit(self, data_dict: Dict[str, np.ndarray]) -> 'ProcrustesAnalyzer':
        """Fit Procrustes on two PCoA coordinate matrices."""
        if len(data_dict) != 2:
            raise ValueError("Procrustes requires exactly two coordinate matrices")
        keys = list(data_dict.keys())
        mtx1, mtx2 = data_dict[keys[0]], data_dict[keys[1]]

        try:
            mtx1_transformed, mtx2_transformed, discrepancy = procrustes(mtx1, mtx2)
        except Exception:
            self._result = {'statistic': 0.0, 'residuals': 0.0, 'transformed_coords': {}}
            return self

        self._result = {
            'statistic': discrepancy,
            'residuals': float(np.sum((mtx1_transformed - mtx2_transformed) ** 2)),
            'transformed_coords': {keys[0]: mtx1_transformed, keys[1]: mtx2_transformed},
        }
        return self

    def fit_transform(self, data_dict: Dict[str, np.ndarray]) -> Dict:
        """Fit and return results."""
        self.fit(data_dict)
        return self.get_results()

    def transform(self, data_dict: Dict[str, np.ndarray]) -> Dict:
        return self._result or {}

    def get_loadings(self) -> pd.DataFrame:
        raise NotImplementedError("Procrustes does not produce loadings")

    def get_results(self) -> Dict:
        result = self._result or {}
        result['method'] = 'procrustes'
        return result