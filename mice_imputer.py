from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

import math
import warnings

import numpy as np

try:
    import polars as pl
except Exception as e:
    raise ImportError("polars is required for MiceImputer") from e

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, log_loss, accuracy_score
except Exception as e:
    raise ImportError("scikit-learn is required for MiceImputer") from e

try:
    import lightgbm as lgb
except Exception as e:
    raise ImportError("lightgbm is required for MiceImputer") from e

try:
    import joblib
except Exception as e:
    raise ImportError("joblib is required for save/load") from e


# ---------- Utilities ----------


def _as_numpy_2d(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    return df.select(cols).to_numpy()


def _is_numeric(series: pl.Series) -> bool:
    return pl.datatypes.is_numeric(series.dtype)


def _safe_mode(values: Iterable[Any]) -> Any:
    arr = list(v for v in values if v is not None)
    if not arr:
        return None
    # tie-breaking deterministic with stable sort of values stringified
    uniq, counts = np.unique(np.array(arr, dtype=object), return_counts=True)
    return uniq[int(np.argmax(counts))]


def _series_null_mask(s: pl.Series) -> np.ndarray:
    return s.is_null().to_numpy()


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, a_min=-0.999999999, a_max=None))


def _safe_expm1(x: np.ndarray) -> np.ndarray:
    return np.expm1(x)


def _one_hot_encode(series: pl.Series) -> Tuple[List[str], pl.DataFrame, List[Any]]:
    # Returns (column_names, encoded_df, categories)
    cats = series.unique(drop_nulls=True).to_list()
    cats_sorted = sorted([c for c in cats if c is not None], key=lambda z: str(z))
    cols = [f"{series.name}__{c}" for c in cats_sorted]
    enc = pl.concat(
        [
            pl.when(series == c).then(1).otherwise(0).alias(col)
            for c, col in zip(cats_sorted, cols)
        ],
        how="horizontal",
    )
    return cols, enc, cats_sorted


def _inverse_one_hot(
    row_vals: np.ndarray, cats: List[Any], rng: np.random.Generator
) -> Any:
    # If multiple 1s or all 0s, pick by probability proportional to values (or uniform if all equal)
    if row_vals.sum() == 1:
        idx = int(np.argmax(row_vals))
        return cats[idx]
    # normalize
    probs = row_vals.astype(float)
    if probs.sum() <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()
    idx = rng.choice(np.arange(len(probs)), p=probs)
    return cats[idx]


def _train_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal["regression", "classification"],
    params: Dict[str, Any],
    random_state: Optional[int],
) -> Any:
    params = dict(params or {})
    if task == "regression":
        model = lgb.LGBMRegressor(random_state=random_state, **params)
    else:
        # Handle binary or multiclass automatically
        model = lgb.LGBMClassifier(random_state=random_state, **params)
    model.fit(X, y)
    return model


def _pmm_sample(
    y_pred_missing: np.ndarray,
    y_pred_observed: np.ndarray,
    y_obs: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    PMM donor selection: find k nearest observed predictions and sample their observed values.
    """
    if len(y_obs) == 0:
        return y_pred_missing.copy()
    nbrs = NearestNeighbors(n_neighbors=min(k, len(y_obs)), algorithm="auto")
    nbrs.fit(y_pred_observed.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(
        y_pred_missing.reshape(-1, 1), return_distance=True
    )
    out = np.empty_like(y_pred_missing, dtype=float)
    for i in range(len(y_pred_missing)):
        donor_ix = rng.choice(indices[i])
        out[i] = y_obs[donor_ix]
    return out


def _enforce_range(
    values: np.ndarray, bounds: Optional[Tuple[Optional[float], Optional[float]]]
) -> np.ndarray:
    if not bounds:
        return values
    lo, hi = bounds
    if lo is not None:
        values = np.maximum(values, lo)
    if hi is not None:
        values = np.minimum(values, hi)
    return values


def _r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _logloss_safe(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, proba, labels=np.unique(y_true)))
    except Exception:
        return float("nan")


def _accuracy_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(accuracy_score(y_true, y_pred))
    except Exception:
        return float("nan")


@dataclass
class ColumnConfig:
    col_type: Literal["continuous", "categorical", "ordinal"]
    pmm: bool = False
    pmm_k: int = 5
    range: Optional[Tuple[Optional[float], Optional[float]]] = None
    order: Optional[List[Any]] = None
    log_transform: bool = False


class MiceImputer:
    def __init__(
        self,
        metadata: Optional[Dict[Union[str, Tuple[str, ...]], Dict[str, Any]]] = None,
        n_iterations: int = 5,
        lgbm_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        convergence_threshold: float = 1e-3,
        max_predictor_missing_rate: float = 0.95,
    ) -> None:
        self.metadata_raw = metadata or {}
        self.n_iterations = int(n_iterations)
        self.lgbm_params = lgbm_params or {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
        }
        self.random_state = random_state
        self.convergence_threshold = float(convergence_threshold)
        self.max_predictor_missing_rate = float(max_predictor_missing_rate)

        self.models_: Dict[str, Any] = {}
        self.config_: Dict[str, ColumnConfig] = {}
        self.onehot_groups_: Dict[
            str, Tuple[List[str], List[Any]]
        ] = {}  # original_name -> (cols, cats)
        self.columns_to_impute_: List[str] = []
        self.predictor_mapping_: Dict[str, List[str]] = {}
        self.diagnostics_: Dict[str, Any] = {}
        self.fitted_: bool = False
        self._rng = _rng_from_seed(random_state)

    # ----- Public API -----

    def fit(
        self,
        X: pl.DataFrame,
        columns_to_impute: Optional[List[str]] = None,
        predictor_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> "MiceImputer":
        df = X.clone()
        self._prepare_metadata(df)
        df_proc = self._preprocess_data(df)
        self._configure_imputation(df_proc, columns_to_impute, predictor_mapping)

        # Warm-start: simple initialization for missing targets
        df_imp = self._initialize_missing(df_proc)

        # Iterative chained equations
        prev_imputed_snapshot = df_imp.select(self.columns_to_impute_).to_numpy()
        changes: List[float] = []
        for it in range(self.n_iterations):
            df_imp = self._fit_iteration(df_imp, it)
            curr = df_imp.select(self.columns_to_impute_).to_numpy()
            delta = float(np.nanmean((curr - prev_imputed_snapshot) ** 2)) ** 0.5
            changes.append(delta)
            prev_imputed_snapshot = curr
            if delta < self.convergence_threshold:
                break

        # Post-process for output and diagnostics
        df_out = self._postprocess_data(df_imp)
        self.diagnostics_ = {
            "iterations_run": len(changes),
            "converged": changes[-1] < self.convergence_threshold if changes else True,
            "per_iter_delta": changes,
            "column_metrics": getattr(self, "_column_metrics", {}),
            "warnings": getattr(self, "_warnings", []),
        }
        self.fitted_ = True
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")
        df = X.clone()
        df_proc = self._preprocess_data(df, expect_known_onehots=True)
        df_imp = self._impute_with_trained_models(df_proc)
        return self._postprocess_data(df_imp)

    def fit_transform(self, X: pl.DataFrame, **fit_params: Any) -> pl.DataFrame:
        return self.fit(X, **fit_params).transform(X)

    def save(self, filepath: str) -> None:
        payload = {
            "models_": self.models_,
            "config_": self.config_,
            "onehot_groups_": self.onehot_groups_,
            "columns_to_impute_": self.columns_to_impute_,
            "predictor_mapping_": self.predictor_mapping_,
            "diagnostics_": self.diagnostics_,
            "n_iterations": self.n_iterations,
            "lgbm_params": self.lgbm_params,
            "random_state": self.random_state,
            "convergence_threshold": self.convergence_threshold,
            "max_predictor_missing_rate": self.max_predictor_missing_rate,
            "metadata_raw": self.metadata_raw,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MiceImputer":
        payload = joblib.load(filepath)
        obj = cls(
            metadata=payload.get("metadata_raw"),
            n_iterations=payload.get("n_iterations"),
            lgbm_params=payload.get("lgbm_params"),
            random_state=payload.get("random_state"),
            convergence_threshold=payload.get("convergence_threshold"),
            max_predictor_missing_rate=payload.get("max_predictor_missing_rate"),
        )
        obj.models_ = payload["models_"]
        obj.config_ = payload["config_"]
        obj.onehot_groups_ = payload["onehot_groups_"]
        obj.columns_to_impute_ = payload["columns_to_impute_"]
        obj.predictor_mapping_ = payload["predictor_mapping_"]
        obj.diagnostics_ = payload["diagnostics_"]
        obj.fitted_ = True
        return obj

    # ----- Internal helpers -----

    def _prepare_metadata(self, df: pl.DataFrame) -> None:
        # Parse metadata into ColumnConfig per atomic column and resolve one-hot groups
        self.config_.clear()
        self.onehot_groups_.clear()

        # First, handle explicit onehot groups in metadata (tuple keys)
        for key, meta in self.metadata_raw.items():
            if isinstance(key, tuple):
                original_name = meta.get("original_name") or "cat"
                # Validate that all columns exist
                missing = [c for c in key if c not in df.columns]
                if missing:
                    raise ValueError(f"One-hot group missing columns: {missing}")
                self.onehot_groups_[original_name] = (
                    list(key),
                    [],
                )  # categories to be filled later
                # Store config for the reconstructed categorical variable (not for the one-hot columns)
                cfg = ColumnConfig(
                    col_type="categorical",
                    pmm=bool(meta.get("pmm", False)),
                    pmm_k=int(meta.get("pmm_k", 5)),
                    range=None,
                    order=meta.get("order"),
                    log_transform=bool(meta.get("log_transform", False)),
                )
                self.config_[original_name] = cfg

        # Then, atomic columns
        for col in df.columns:
            # Skip columns that belong to a declared onehot group
            if any(col in cols for cols, _ in self.onehot_groups_.values()):
                continue
            meta = self.metadata_raw.get(col)
            if meta is None:
                # infer
                s = df[col]
                inferred = self._infer_column_metadata(col, s)
                self.config_[col] = ColumnConfig(**inferred)
            else:
                col_type = meta.get("type", "continuous")
                if col_type not in ("continuous", "categorical", "ordinal"):
                    raise ValueError(f"Unsupported type for {col}: {col_type}")
                self.config_[col] = ColumnConfig(
                    col_type=col_type,
                    pmm=bool(meta.get("pmm", False)),
                    pmm_k=int(meta.get("pmm_k", 5)),
                    range=meta.get("range"),
                    order=meta.get("order"),
                    log_transform=bool(meta.get("log_transform", False)),
                )

    def _preprocess_data(
        self, df: pl.DataFrame, expect_known_onehots: bool = False
    ) -> pl.DataFrame:
        out = df.clone()

        # Reconstruct categorical variables from one-hot groups into a single column
        for orig, (cols, cats) in self.onehot_groups_.items():
            # Validate binary 0/1
            for c in cols:
                if not _is_numeric(out[c]):
                    raise ValueError(f"One-hot column {c} must be numeric 0/1")
                bad = out[c].drop_nulls().filter(~pl.col(c).is_in([0, 1])).height > 0
                if bad:
                    warnings.warn(
                        f"One-hot column {c} has non-binary values; will coerce by rounding."
                    )
                    out = out.with_columns(pl.col(c).round(0).clip(0, 1))

            # Build categories from column suffixes if not set
            if not cats:
                new_cats = [c.split("__")[-1] if "__" in c else c for c in cols]
                self.onehot_groups_[orig] = (cols, new_cats)

            # Create reconstructed column with None for rows with all null one-hot inputs
            vals = out.select(cols).to_numpy()
            mask_all_null = np.all(np.isnan(vals), axis=1)
            recon = []
            for i in range(vals.shape[0]):
                row = vals[i]
                if mask_all_null[i]:
                    recon.append(None)
                else:
                    recon.append(
                        _inverse_one_hot(
                            np.nan_to_num(row, nan=0),
                            self.onehot_groups_[orig][1],
                            self._rng,
                        )
                    )
            out = out.with_columns(pl.Series(name=orig, values=recon))

        # Apply log transform as configured (in-place on reconstructed or original columns)
        for col, cfg in self.config_.items():
            if cfg.log_transform and col in out.columns:
                s = out[col]
                if _is_numeric(s):
                    arr = s.to_numpy()
                    arr_t = _safe_log1p(np.where(np.isnan(arr), np.nan, arr))
                    out = out.with_columns(pl.Series(name=col, values=arr_t))
                else:
                    warnings.warn(
                        f"log_transform requested for non-numeric column {col}; ignoring."
                    )

        return out

    def _configure_imputation(
        self,
        df: pl.DataFrame,
        columns_to_impute: Optional[List[str]],
        predictor_mapping: Optional[Dict[str, List[str]]],
    ) -> None:
        # Decide columns to impute: from args or infer by nulls
        if columns_to_impute is None:
            self.columns_to_impute_ = [
                c for c in df.columns if df[c].is_null().any() and c in self.config_
            ]
        else:
            self.columns_to_impute_ = list(columns_to_impute)

        # Default predictors: all other columns (excluding one-hot primitive columns)
        default_predictors = [c for c in df.columns if c in self.config_]
        mapping = {}
        for tgt in self.columns_to_impute_:
            if predictor_mapping and tgt in predictor_mapping:
                pr = [c for c in predictor_mapping[tgt] if c in df.columns]
            else:
                pr = [c for c in default_predictors if c != tgt]
            # Drop predictors with extreme missingness to avoid degenerate training sets
            pr = [
                c
                for c in pr
                if (1.0 - df[c].is_null().mean())
                >= (1 - self.max_predictor_missing_rate)
            ]
            mapping[tgt] = pr
        self.predictor_mapping_ = mapping

        # Imputation order: most missing first
        miss_rate = {c: float(df[c].is_null().mean()) for c in self.columns_to_impute_}
        self.columns_to_impute_.sort(key=lambda c: (-miss_rate[c], c))

    def _initialize_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.clone()
        for col in self.columns_to_impute_:
            cfg = self.config_[col]
            s = out[col]
            mask = _series_null_mask(s)
            if not mask.any():
                continue
            non_null = s.drop_nulls()
            if non_null.is_empty():
                # no data to infer; leave as nulls
                continue
            if cfg.col_type == "continuous":
                val = float(non_null.mean())
                if math.isnan(val):
                    # fallback
                    val = float(non_null.to_numpy()[0])
                init_vals = np.where(mask, val, s.to_numpy())
            else:
                val = _safe_mode(non_null.to_list())
                init_vals = np.where(mask, val, s.to_numpy(object))
            out = out.with_columns(pl.Series(name=col, values=init_vals))
        return out

    def _fit_iteration(self, data: pl.DataFrame, iteration: int) -> pl.DataFrame:
        out = data.clone()
        rng = self._rng
        col_metrics: Dict[str, Dict[str, Any]] = getattr(self, "_column_metrics", {})
        self.models_.clear()

        for target in self.columns_to_impute_:
            cfg = self.config_[target]
            predictors = [
                c for c in self.predictor_mapping_[target] if c in out.columns
            ]

            # Partition observed/missing rows
            y = out[target]
            mask_missing = _series_null_mask(y)
            if mask_missing.sum() == 0:
                continue

            X_obs = _as_numpy_2d(out.filter(~pl.Series(mask_missing)), predictors)
            X_mis = _as_numpy_2d(out.filter(pl.Series(mask_missing)), predictors)

            y_obs_series = out.filter(~pl.Series(mask_missing))[target]
            if cfg.col_type == "continuous":
                y_obs = y_obs_series.to_numpy().astype(float)
                task = "regression"
            else:
                # categorical/ordinal as labels (ordered if ordinal with provided order)
                if cfg.col_type == "ordinal" and cfg.order is not None:
                    cat_to_idx = {c: i for i, c in enumerate(cfg.order)}
                    y_obs = np.array(
                        [
                            cat_to_idx.get(v, np.nan)
                            for v in y_obs_series.to_list()
                            if v is not None
                        ]
                    )
                    # Need to make sure we align X_obs rows
                    not_nan_mask = ~np.isnan(y_obs)
                    X_obs = X_obs[not_nan_mask]
                    y_obs = y_obs[not_nan_mask].astype(int)
                else:
                    y_obs = np.array(y_obs_series.to_list(), dtype=object)
                    # encode to indices
                    cats = sorted(np.unique(y_obs))
                    cat_to_idx = {c: i for i, c in enumerate(cats)}
                    y_obs = np.array([cat_to_idx[v] for v in y_obs])

                task = "classification"

            # Train model; if insufficient data, skip with simple fill
            if len(y_obs) < 5 or X_obs.shape[0] < 5:
                warnings.warn(
                    f"Insufficient training data for {target}; using simple fill."
                )
                if cfg.col_type == "continuous":
                    fill_val = float(np.nanmean(out[target].to_numpy()))
                    imputed = np.full(X_mis.shape[0], fill_val, dtype=float)
                    if cfg.range:
                        imputed = _enforce_range(imputed, cfg.range)
                else:
                    fill_val = _safe_mode(out[target].to_list())
                    imputed = np.array([fill_val] * X_mis.shape[0], dtype=object)
                # Update
                all_vals = out[target].to_numpy(object)
                all_vals[mask_missing] = imputed
                out = out.with_columns(pl.Series(name=target, values=all_vals))
                continue

            # Train LightGBM
            model = _train_lgbm(
                X_obs,
                y_obs,
                task=task,
                params=self.lgbm_params,
                random_state=self.random_state,
            )
            self.models_[target] = model

            # Evaluate on a small holdout for diagnostics (optional)
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_obs,
                    y_obs,
                    test_size=min(0.2, max(0.1, 1.0 / max(10, len(y_obs)))),
                    random_state=self.random_state,
                    shuffle=True,
                )
                model_hold = _train_lgbm(
                    X_tr,
                    y_tr,
                    task=task,
                    params=self.lgbm_params,
                    random_state=self.random_state,
                )
                if task == "regression":
                    y_hat = model_hold.predict(X_te)
                    col_metrics[target] = {
                        "r2_score": _r2_safe(y_te, y_hat),
                        "pmm_used": bool(cfg.pmm),
                    }
                else:
                    proba = model_hold.predict_proba(X_te)
                    y_hat = np.argmax(proba, axis=1)
                    col_metrics[target] = {
                        "accuracy": _accuracy_safe(y_te, y_hat),
                        "log_loss": _logloss_safe(y_te, proba),
                        "pmm_used": False,
                    }
            except Exception:
                pass
            self._column_metrics = col_metrics

            # Predict missing
            if task == "regression":
                y_pred_obs = model.predict(X_obs)
                y_pred_mis = model.predict(X_mis)
                if cfg.pmm:
                    donors = _pmm_sample(
                        y_pred_mis, y_pred_obs, y_obs, k=cfg.pmm_k, rng=rng
                    )
                    imputed = donors
                else:
                    imputed = y_pred_mis
                if cfg.range:
                    imputed = _enforce_range(imputed, cfg.range)
            else:
                proba = model.predict_proba(X_mis)
                # Sample according to probabilities
                imputed_idx = np.array(
                    [
                        rng.choice(np.arange(proba.shape[1]), p=proba[i])
                        for i in range(proba.shape[0])
                    ]
                )
                if cfg.col_type == "ordinal" and cfg.order is not None:
                    cats = cfg.order
                else:
                    # If we learned cats earlier, attempt to recover
                    if hasattr(model, "classes_"):
                        cats = list(model.classes_)
                    else:
                        cats = list(range(proba.shape[1]))
                imputed = np.array([cats[i] for i in imputed_idx], dtype=object)

            # Update target in out
            all_vals = out[target].to_numpy(object)
            all_vals[mask_missing] = imputed
            out = out.with_columns(pl.Series(name=target, values=all_vals))

        return out

    def _impute_with_trained_models(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.clone()
        rng = self._rng
        for target in self.columns_to_impute_:
            if target not in out.columns:
                continue
            cfg = self.config_[target]
            mask_missing = _series_null_mask(out[target])
            if mask_missing.sum() == 0:
                continue
            predictors = [
                c for c in self.predictor_mapping_[target] if c in out.columns
            ]
            X_mis = _as_numpy_2d(out.filter(pl.Series(mask_missing)), predictors)

            model = self.models_.get(target)
            if model is None:
                # fallback: simple fill
                s = out[target]
                if cfg.col_type == "continuous":
                    fill_val = float(s.drop_nulls().mean())
                    imputed = np.full(X_mis.shape[0], fill_val, dtype=float)
                    if cfg.range:
                        imputed = _enforce_range(imputed, cfg.range)
                else:
                    fill_val = _safe_mode(s.drop_nulls().to_list())
                    imputed = np.array([fill_val] * X_mis.shape[0], dtype=object)
            else:
                if cfg.col_type == "continuous":
                    imputed = model.predict(X_mis)
                    if cfg.range:
                        imputed = _enforce_range(imputed, cfg.range)
                else:
                    proba = model.predict_proba(X_mis)
                    imputed_idx = np.array(
                        [
                            rng.choice(np.arange(proba.shape[1]), p=proba[i])
                            for i in range(proba.shape[0])
                        ]
                    )
                    if hasattr(model, "classes_"):
                        cats = list(model.classes_)
                    else:
                        cats = list(range(proba.shape[1]))
                    imputed = np.array([cats[i] for i in imputed_idx], dtype=object)

            all_vals = out[target].to_numpy(object)
            all_vals[mask_missing] = imputed
            out = out.with_columns(pl.Series(name=target, values=all_vals))
        return out

    def _postprocess_data(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.clone()

        # Inverse log transforms
        for col, cfg in self.config_.items():
            if cfg.log_transform and col in out.columns:
                s = out[col]
                if _is_numeric(s):
                    arr = s.to_numpy()
                    inv = _safe_expm1(np.where(np.isnan(arr), np.nan, arr))
                    out = out.with_columns(pl.Series(name=col, values=inv))

        # Re-encode reconstructed categoricals back to one-hots if groups exist
        for orig, (cols, cats) in self.onehot_groups_.items():
            if orig not in out.columns:
                continue
            s = out[orig]
            # Build mapping from category to one-hot row
            oh_data = []
            for c in cats:
                oh_data.append(
                    pl.when(s == c)
                    .then(1)
                    .otherwise(0)
                    .alias(
                        [
                            c
                            for c in cols
                            if c.endswith(str(c)) or c.split("__")[-1] == str(c)
                        ][0]
                        if cols
                        else f"{orig}__{c}"
                    )
                )
            if oh_data:
                enc_df = pl.concat(oh_data, how="horizontal")
                out = out.drop(orig).hstack(enc_df, in_place=False)

        return out

    # ----- Defaults & validation -----

    def _infer_column_metadata(self, col: str, data: pl.Series) -> Dict[str, Any]:
        # Numeric?
        if _is_numeric(data):
            unique_non_null = data.drop_nulls().unique().len()
            if unique_non_null <= 2:
                return {
                    "col_type": "categorical",
                    "pmm": False,
                    "pmm_k": 5,
                    "range": None,
                    "order": None,
                    "log_transform": False,
                }
            if unique_non_null < 10:
                return {
                    "col_type": "ordinal",
                    "pmm": False,
                    "pmm_k": 5,
                    "range": None,
                    "order": None,
                    "log_transform": False,
                }
            return {
                "col_type": "continuous",
                "pmm": True,
                "pmm_k": 5,
                "range": None,
                "order": None,
                "log_transform": False,
            }
        else:
            return {
                "col_type": "categorical",
                "pmm": False,
                "pmm_k": 5,
                "range": None,
                "order": None,
                "log_transform": False,
            }

    # Expose diagnostics
    @property
    def diagnostics_(self) -> Dict[str, Any]:
        return self._diagnostics

    @diagnostics_.setter
    def diagnostics_(self, value: Dict[str, Any]) -> None:
        self._diagnostics = value
