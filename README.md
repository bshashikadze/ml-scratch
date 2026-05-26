# ml-scratch

Classical machine learning methods implemented from scratch in Python, with a focus on high-dimensional low-sample-size regimes encountered in proteomics.

## Methods

| Method | Module | Status |
|---|---|---|
| PCA (eigendecomposition) | `mlscratch.decomposition.PCAEig` | done |
| PCA (SVD) | `mlscratch.decomposition.PCASVD` | in progress |
| Linear Regression (OLS) | — | planned |
| Ridge / Lasso / Elastic Net | — | planned |
| LIMMA-style linear modeling | — | planned |
| Decision Trees | — | planned |
| Random Forest | — | planned |
| XGBoost | — | planned |
| t-SNE | — | planned |

## Installation

```bash
git clone https://github.com/bshashikadze/ml-scratch.git
cd ml-scratch
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```python
from mlscratch.decomposition import PCAEig

pca = PCAEig(n_components=2)
pca.fit(X)
Z = pca.transform(X)
```

## Design

- **NumPy + `np.linalg` only.** `sklearn` is used only for validation in tests and notebooks; the package itself has no scikit-learn dependency.
- **scikit-learn-style API.** `fit`, `transform`, `fit_transform`, `inverse_transform`. Fitted attributes end with `_`.
- **Type hints throughout.**

## License

MIT