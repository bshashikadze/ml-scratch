"""PCA via eigendecomposition of the covariance matrix."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PCAEig:
    """PCA implemented via eigendecomposition of the covariance matrix.

    Parameters
    ----------
    n_components
        Number of principal components to keep.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal components, stored as rows. Sorted by descending variance.
    explained_variance_ : ndarray of shape (n_components,)
        Variance along each principal component (the top eigenvalues).
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance captured by each principal component.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean learned at fit, used to center data at transform time.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def __repr__(self) -> str:
        return f"PCAEig(n_components={self.n_components})"

    def fit(self, X: NDArray[np.floating]) -> "PCAEig":
        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} cannot exceed "
                f"n_features={n_features}"
            )

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov = (X_centered.T @ X_centered) / (n_samples - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        total_variance = eigenvalues.sum()

        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.components_ = eigenvectors[:, : self.n_components].T

        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        return Z @ self.components_ + self.mean_