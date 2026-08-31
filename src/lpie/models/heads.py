"""Direct-horizon GBDT heads and the baseline ladder.

The compounded hazard can drift from the empirical horizon rate if the one-step
model is slightly miscalibrated — errors compound multiplicatively over twelve
steps. So each horizon also gets a direct binary GBDT, and the two are blended
by a stacked meta-learner (see `ensemble.py`).

Three GBDT families are trained per head because their errors are decorrelated,
which is the only thing that makes stacking pay: LightGBM (native NaN and
categorical handling, both essential here), XGBoost (different regularisation and
split algorithm), CatBoost (ordered boosting, less prone to categorical target
leakage, strong out-of-box calibration).

Class imbalance is handled with `scale_pos_weight`, never SMOTE. Synthesising
minority rows inside a temporally ordered panel fabricates loan-months that never
existed, breaks the point-in-time guarantee, and reliably degrades calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import ModelNotLoadedError, PredictionError
from lpie.core.logging import get_logger

log = get_logger(__name__)

ALGORITHMS = ("lightgbm", "xgboost", "catboost")


@dataclass
class HeadArtifact:
    head: str
    algorithm: str
    model: Any
    feature_names: list[str]
    categorical_features: list[str] = field(default_factory=list)
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    best_iteration: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    train_window: str = ""
    scale_pos_weight: float | None = None


def _pin_categories(
    X: pd.DataFrame,
    feature_names: list[str],
    categorical_features: list[str],
    levels: dict[str, list[str]] | None = None,
    *,
    encode_numeric: bool = False,
) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise PredictionError(
            f"Feature matrix is missing {len(missing)} declared feature(s)",
            details={"missing": missing[:20]},
        )
    out = X[feature_names].copy()
    for c in categorical_features:
        if c not in out.columns:
            continue
        cats = (levels or {}).get(c)
        col = pd.Categorical(out[c].astype("object"), categories=cats) if cats else pd.Categorical(out[c].astype("object"))
        out[c] = col.codes.astype("float64") if encode_numeric else col
        if encode_numeric:
            out.loc[out[c] < 0, c] = np.nan
    for c in out.columns:
        if c not in categorical_features and not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fit_category_levels(X: pd.DataFrame, categorical_features: list[str]) -> dict[str, list[str]]:
    return {
        c: sorted(X[c].dropna().astype(str).unique().tolist())
        for c in categorical_features
        if c in X.columns
    }


# --------------------------------------------------------------------------- #
def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    head: str,
    feature_names: list[str],
    categorical_features: list[str],
    category_levels: dict[str, list[str]],
    valid: tuple[pd.DataFrame, pd.Series] | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 60,
    scale_pos_weight: float | None = None,
    settings: Settings | None = None,
) -> HeadArtifact:
    import lightgbm as lgb

    s = settings or get_settings()
    X_fit = _pin_categories(X, feature_names, categorical_features, category_levels)
    cats = [c for c in categorical_features if c in X_fit.columns]

    base = {
        "objective": "binary",
        "metric": ["average_precision", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": s.seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": int(s.get("runtime.duckdb_threads", 4)),
    }
    if scale_pos_weight:
        base["scale_pos_weight"] = float(scale_pos_weight)
    base.update(params or {})

    train_set = lgb.Dataset(X_fit, label=y.to_numpy(), categorical_feature=cats, free_raw_data=False)
    valid_sets, callbacks = None, [lgb.log_evaluation(period=0)]
    if valid is not None:
        Xv, yv = valid
        valid_sets = [
            lgb.Dataset(
                _pin_categories(Xv, feature_names, categorical_features, category_levels),
                label=yv.to_numpy(), categorical_feature=cats, reference=train_set, free_raw_data=False,
            )
        ]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    booster = lgb.train(base, train_set, num_boost_round=num_boost_round,
                        valid_sets=valid_sets, callbacks=callbacks)
    return HeadArtifact(
        head=head, algorithm="lightgbm", model=booster, feature_names=list(feature_names),
        categorical_features=cats, category_levels=category_levels,
        best_iteration=booster.best_iteration or booster.current_iteration(),
        scale_pos_weight=scale_pos_weight,
    )


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    head: str,
    feature_names: list[str],
    categorical_features: list[str],
    category_levels: dict[str, list[str]],
    valid: tuple[pd.DataFrame, pd.Series] | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 60,
    scale_pos_weight: float | None = None,
    settings: Settings | None = None,
) -> HeadArtifact:
    import xgboost as xgb

    s = settings or get_settings()
    X_fit = _pin_categories(X, feature_names, categorical_features, category_levels, encode_numeric=True)

    base = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "seed": s.seed,
        "nthread": int(s.get("runtime.duckdb_threads", 4)),
        "tree_method": "hist",
    }
    if scale_pos_weight:
        base["scale_pos_weight"] = float(scale_pos_weight)
    base.update(params or {})

    dtrain = xgb.DMatrix(X_fit, label=y.to_numpy(), missing=np.nan)
    evals, kwargs = [], {}
    if valid is not None:
        Xv, yv = valid
        dvalid = xgb.DMatrix(
            _pin_categories(Xv, feature_names, categorical_features, category_levels, encode_numeric=True),
            label=yv.to_numpy(), missing=np.nan,
        )
        evals = [(dvalid, "valid")]
        kwargs["early_stopping_rounds"] = early_stopping_rounds

    booster = xgb.train(base, dtrain, num_boost_round=num_boost_round, evals=evals,
                        verbose_eval=False, **kwargs)
    return HeadArtifact(
        head=head, algorithm="xgboost", model=booster, feature_names=list(feature_names),
        categorical_features=list(categorical_features), category_levels=category_levels,
        best_iteration=getattr(booster, "best_iteration", None),
        scale_pos_weight=scale_pos_weight,
    )


def train_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    head: str,
    feature_names: list[str],
    categorical_features: list[str],
    category_levels: dict[str, list[str]],
    valid: tuple[pd.DataFrame, pd.Series] | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 60,
    scale_pos_weight: float | None = None,
    settings: Settings | None = None,
) -> HeadArtifact:
    from catboost import CatBoostClassifier, Pool

    s = settings or get_settings()

    def prep(frame: pd.DataFrame) -> pd.DataFrame:
        out = _pin_categories(frame, feature_names, categorical_features, category_levels)
        for c in categorical_features:
            if c in out.columns:
                out[c] = out[c].astype("object").where(out[c].notna(), "Unknown").astype(str)
        return out

    X_fit = prep(X)
    cats = [c for c in categorical_features if c in X_fit.columns]

    base = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "learning_rate": 0.06,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": s.seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": int(s.get("runtime.duckdb_threads", 4)),
        "iterations": num_boost_round,
    }
    if scale_pos_weight:
        base["scale_pos_weight"] = float(scale_pos_weight)
    base.update(params or {})

    model = CatBoostClassifier(**base)
    train_pool = Pool(X_fit, label=y.to_numpy(), cat_features=cats)
    eval_pool = None
    if valid is not None:
        Xv, yv = valid
        eval_pool = Pool(prep(Xv), label=yv.to_numpy(), cat_features=cats)

    model.fit(train_pool, eval_set=eval_pool,
              early_stopping_rounds=early_stopping_rounds if eval_pool else None, verbose=False)
    return HeadArtifact(
        head=head, algorithm="catboost", model=model, feature_names=list(feature_names),
        categorical_features=cats, category_levels=category_levels,
        best_iteration=int(model.get_best_iteration()) if eval_pool else None,
        scale_pos_weight=scale_pos_weight,
    )


TRAINERS = {
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "catboost": train_catboost,
}


def predict_head(artifact: HeadArtifact, X: pd.DataFrame) -> np.ndarray:
    """Uniform prediction interface across the three GBDT libraries."""
    if artifact is None or artifact.model is None:
        raise ModelNotLoadedError(
            "Head artifact is not loaded", details={"head": getattr(artifact, "head", "unknown")}
        )
    algo = artifact.algorithm
    if algo == "lightgbm":
        X_pred = _pin_categories(X, artifact.feature_names, artifact.categorical_features,
                                 artifact.category_levels)
        return np.asarray(
            artifact.model.predict(X_pred, num_iteration=artifact.best_iteration), dtype="float64"
        )
    if algo == "xgboost":
        import xgboost as xgb

        X_pred = _pin_categories(X, artifact.feature_names, artifact.categorical_features,
                                 artifact.category_levels, encode_numeric=True)
        dmat = xgb.DMatrix(X_pred, missing=np.nan)
        kwargs = {}
        if artifact.best_iteration is not None:
            kwargs["iteration_range"] = (0, int(artifact.best_iteration) + 1)
        return np.asarray(artifact.model.predict(dmat, **kwargs), dtype="float64")
    if algo == "catboost":
        X_pred = _pin_categories(X, artifact.feature_names, artifact.categorical_features,
                                 artifact.category_levels)
        for c in artifact.categorical_features:
            if c in X_pred.columns:
                X_pred[c] = X_pred[c].astype("object").where(X_pred[c].notna(), "Unknown").astype(str)
        return np.asarray(artifact.model.predict_proba(X_pred)[:, 1], dtype="float64")
    raise PredictionError(f"Unknown algorithm '{algo}'")


# --------------------------------------------------------------------------- #
# Baseline ladder B0..B5
# --------------------------------------------------------------------------- #
def baseline_prior(y_train: pd.Series, n_predict: int) -> np.ndarray:
    """B0 — the trivial floor: predict the training base rate for everyone."""
    return np.full(n_predict, float(pd.to_numeric(y_train, errors="coerce").mean()), dtype="float64")


def baseline_current_state(
    y_train: pd.Series, status_train: pd.Series, status_predict: pd.Series
) -> np.ndarray:
    """B1 — the "do you beat a spreadsheet?" bar: empirical rate by current_status.

    This is the baseline that matters. On next-state prediction it alone reaches
    ~0.94 accuracy, which is exactly why the headline accuracy number is
    reported alongside the active-conditional one.
    """
    y = pd.to_numeric(y_train, errors="coerce")
    rates = y.groupby(status_train.astype("object")).mean()
    overall = float(y.mean())
    return (
        status_predict.astype("object").map(rates).astype("float64").fillna(overall).to_numpy()
    )


def baseline_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_predict: pd.DataFrame,
    *,
    feature_names: list[str],
    categorical_features: list[str],
    settings: Settings | None = None,
) -> np.ndarray:
    """B2 — regularised logistic regression on one-hot + median-imputed numerics."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    s = settings or get_settings()
    numeric = [c for c in feature_names if c not in categorical_features]
    cats = [c for c in categorical_features if c in X_train.columns]

    pipeline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                          ("scale", StandardScaler())]), numeric),
                        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), cats),
                    ],
                    remainder="drop",
                ),
            ),
            ("clf", LogisticRegression(max_iter=400, C=1.0, solver="lbfgs", random_state=s.seed)),
        ]
    )
    train = X_train[feature_names].copy()
    predict = X_predict[feature_names].copy()
    for frame in (train, predict):
        for c in cats:
            frame[c] = frame[c].astype("object").where(frame[c].notna(), "Unknown").astype(str)
    pipeline.fit(train, y_train.to_numpy())
    return np.asarray(pipeline.predict_proba(predict)[:, 1], dtype="float64")


def scale_pos_weight_for(y: pd.Series) -> float | None:
    """neg/pos ratio. Returned only when the class is genuinely rare."""
    y_num = pd.to_numeric(y, errors="coerce").dropna()
    pos = float((y_num > 0).sum())
    neg = float((y_num <= 0).sum())
    if pos <= 0 or neg <= 0:
        return None
    ratio = neg / pos
    return round(ratio, 4) if ratio > 4.0 else None
