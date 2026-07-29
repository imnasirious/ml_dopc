"""
xgb_trainer_lib.py
------------------
SVC-style wrapper for XGBoost with (optional) HalvingGridSearchCV/ GridSearchCV.

Goal: write code analogous to SVC:
    xgb_1 = XGBoostGS(use_scaler=False, halving=True, n_jobs=-1)
    xgb_1.fit(X_train_scaled, y_train)
    y_pred = xgb_1.predict(X_test_scaled)

Notes on scaling:
- XGBoost (tree-based) does NOT require feature scaling. Order-based splits are
  invariant to linear scaling. So we default to `use_scaler=False`.
- If your pipeline already does manual scaling to stay analogous with an SVC script,
  keep `use_scaler=False` and pass the scaled arrays. It's harmless either way.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, Dict, Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import accuracy_score

# Enable and import halving search if you want that mode
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

from xgboost import XGBClassifier


class XGBoostGS:
    """
    SVC-like estimator that performs (Halving)GridSearchCV over a Pipeline:
        (Scaler or Identity) -> XGBClassifier

    Parameters
    ----------
    param_grid : dict or None
        Hyperparameter grid for the XGBClassifier (and optionally scaler).
        If None, a sensible default grid is used.
    n_splits : int
        StratifiedKFold splits if `cv` is not provided.
    random_state : int
        Sets `random_state` on XGBClassifier; also used in StratifiedKFold shuffling.
    scoring : str
        Metric for (Halving)GridSearchCV (default "accuracy").
    n_jobs : int
        Parallelism for the search AND for XGBClassifier (set in estimator).
    verbose : int
        Search verbosity.
    use_scaler : bool
        If True, include StandardScaler; if False, identity transformer.
        (Scaling is not required for trees; default False.)
    cv : Optional[PredefinedSplit]
        If provided, overrides StratifiedKFold.
    halving : bool
        If True, use HalvingGridSearchCV; else GridSearchCV.
    halving_kwargs : dict
        Extra kwargs for HalvingGridSearchCV (e.g., resource, min_resources, factor...).
    xgb_kwargs : dict
        Extra kwargs for XGBClassifier (merged with defaults below).
    """

    def __init__(self,
                 param_grid: Optional[Dict[str, Any]] = None,
                 n_splits: int = 5,
                 random_state: int = 212,
                 scoring: str = "accuracy",
                 n_jobs: int = -1,
                 verbose: int = 0,
                 use_scaler: bool = False,
                 cv: Optional[PredefinedSplit] = None,
                 halving: bool = True,
                 halving_kwargs: Optional[Dict[str, Any]] = None,
                 xgb_kwargs: Optional[Dict[str, Any]] = None):
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.random_state = random_state
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.use_scaler = use_scaler
        self.cv = cv
        self.halving = halving
        self.halving_kwargs = halving_kwargs or {}
        self.xgb_kwargs = xgb_kwargs or {}

        # set after .fit()
        self.search_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

    def _default_param_grid(self) -> Dict[str, Any]:
        # Let halving control n_estimators as the "resource" by default
        # If using plain GridSearch, n_estimators stays fixed unless you add it.
        return {
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__gamma": [0, 1],
            "clf__min_child_weight": [1, 5],
            # You may also grid "clf__reg_alpha", "clf__reg_lambda" if needed
        }

    def _make_estimator(self) -> XGBClassifier:
        est = XGBClassifier(
            eval_metric="logloss",
            random_state=self.random_state,
            tree_method="hist",
            verbosity=0,
            n_jobs=self.n_jobs,
            **self.xgb_kwargs,
        )
        return est

    def _make_search(self):
        scaler_step = StandardScaler() if self.use_scaler else FunctionTransformer(validate=False)

        pipe = Pipeline([
            ("scaler", scaler_step),
            ("clf", self._make_estimator()),
        ])

        grid = self.param_grid if self.param_grid is not None else self._default_param_grid()

        cv = self.cv if self.cv is not None else StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        if self.halving:
            # Provide defaults similar to your script if not supplied
            hkw = dict(
                resource="clf__n_estimators",
                min_resources=100,
                max_resources=1000,
                factor=3,
                aggressive_elimination=True,
            )
            hkw.update(self.halving_kwargs)

            search = HalvingGridSearchCV(
                estimator=pipe,
                param_grid=grid,
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=True,
                **hkw,
            )
        else:
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=True,
                return_train_score=False,
            )

        return search

    # ---- SVC-like API ----
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.search_ = self._make_search()
        self.search_.fit(X, y)
        self.best_estimator_ = self.search_.best_estimator_
        self.best_params_ = self.search_.best_params_
        self.best_score_ = self.search_.best_score_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # For binary classification, returns [:, 2] columns (prob of each class)
        return self.best_estimator_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.best_estimator_.score(X, y)

    def save(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Save the whole fitted wrapper (.sav). Optionally save scaler step separately.
        """
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

        if scaler_path:
            try:
                scaler = self.best_estimator_.named_steps["scaler"]
                os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
            except KeyError:
                pass
