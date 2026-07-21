"""Meta-fusion combiners (global least-squares and Ridge) from the notebook's
"meta fusion" comparison (cell 9).

The fitted models were never persisted by the training notebook -- only their
evaluation metrics (meta_fusion_comparison.csv) were saved. This module
refits them from outputs_.../oof_val_predictions.csv, exactly reproducing the
notebook's fitting code:

  - restrict OOF rows to those with all 4 image modalities present
  - global_lstsq_weights: sklearn LinearRegression on the 5 per-modality mus
  - ridge_meta_learner:   sklearn Ridge(alpha=1.0) on the 5 mus + 5 vars

Both are fit ONCE, globally, on the pooled OOF set across folds (not per
fold) -- matching the notebook, which fits once and evaluates per fold.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from model import ALL_MODALITY_KEYS, IMAGE_MODALITY_KEYS

MU_COLS = [f"mu_{k}" for k in ALL_MODALITY_KEYS]
VAR_COLS = [f"var_{k}" for k in ALL_MODALITY_KEYS]


def _restrict_all_present(df):
    mask = pd.Series(True, index=df.index)
    for key in IMAGE_MODALITY_KEYS:
        mask &= df[f"has_{key}"]
    return df[mask].reset_index(drop=True)


def fit_meta_learners(oof_csv_path):
    """Returns (lstsq_model, ridge_model, n_rows_used)."""
    df = pd.read_csv(oof_csv_path)
    restricted = _restrict_all_present(df)

    X_mu = restricted[MU_COLS].values
    X_mu_var = restricted[MU_COLS + VAR_COLS].values
    y = restricted["y_true"].values

    lstsq_model = LinearRegression().fit(X_mu, y)
    ridge_model = Ridge(alpha=1.0).fit(X_mu_var, y)
    return lstsq_model, ridge_model, len(restricted)


def predict_lstsq(model, per_modality):
    x = [[per_modality[k]["hb_pred"] for k in ALL_MODALITY_KEYS]]
    return float(model.predict(x)[0])


def predict_ridge(model, per_modality):
    x = [[per_modality[k]["hb_pred"] for k in ALL_MODALITY_KEYS]
         + [per_modality[k]["hb_std"] ** 2 for k in ALL_MODALITY_KEYS]]
    return float(model.predict(x)[0])


def all_image_modalities_present(per_modality):
    return all(per_modality[k]["present"] for k in IMAGE_MODALITY_KEYS)
