"""
logreg_momentum_lib.py
----------------------
SVC-style wrapper for TRUE logistic regression (single sigmoid) trained with
SGD + momentum (optional Nesterov), tuned via GridSearchCV.

Use it analogously to SVC:

    from logreg_momentum_lib import LogisticMomentumGS

    log_reg_1 = LogisticMomentumGS(
        n_features=X_train_scaled.shape[1],
        use_scaler=False,          # you already scaled like SVC
        n_splits=5, scoring="accuracy", n_jobs=1, verbose=0
    )

    log_reg_1.fit(X_train_scaled, y_train)
    y_pred = log_reg_1.predict(X_test_scaled)
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, Dict, Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from scikeras.wrappers import KerasClassifier
from tensorflow.keras import Input, Model, optimizers, layers


# ---- Core Keras builder: true logistic regression ----
def build_tf_logistic(n_features: int,
                      learning_rate: float = 1e-2,
                      momentum: float = 0.9,
                      nesterov: bool = True):
    inp = Input(shape=(n_features,))
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(inp)
    model = Model(inp, out)
    opt = optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


class LogisticMomentumGS:
    """
    SVC-like estimator running GridSearchCV over a Pipeline:
        (Scaler or Identity) -> KerasClassifier(build_tf_logistic)

    Parameters
    ----------
    n_features : int
    param_grid : dict or None
        NOTE: model-builder params must be prefixed with `model__`, e.g.
            "clf__model__learning_rate", "clf__model__momentum", "clf__model__nesterov"
        KerasClassifier fit params remain unprefixed beyond `clf__`, e.g.
            "clf__epochs", "clf__batch_size"
    n_splits : int
    random_state : int
    scoring : str
    n_jobs : int
    verbose : int
    use_scaler : bool
        If True, include StandardScaler; if False, identity (so you can do manual scaling).
    cv : Optional[PredefinedSplit]
    """

    def __init__(self,
                 n_features: int,
                 param_grid: Optional[Dict[str, Any]] = None,
                 n_splits: int = 5,
                 random_state: int = 212,
                 scoring: str = "accuracy",
                 n_jobs: int = 1,
                 verbose: int = 0,
                 use_scaler: bool = False,
                 cv: Optional[PredefinedSplit] = None):
        self.n_features = n_features
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.random_state = random_state
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.use_scaler = use_scaler
        self.cv = cv

        self.grid_: Optional[GridSearchCV] = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

    def _make_grid(self) -> GridSearchCV:
        scaler_step = StandardScaler() if self.use_scaler else FunctionTransformer(validate=False)

        pipe = Pipeline([
            ("scaler", scaler_step),
            ("clf", KerasClassifier(
                model=build_tf_logistic,
                n_features=self.n_features,
                verbose=0
            )),
        ])

        if self.param_grid is None:
            # IMPORTANT: model-builder args are under `model__...`
            self.param_grid = {
                "clf__model__learning_rate": [1e-2, 5e-3, 1e-3],
                "clf__model__momentum": [0.0, 0.9],
                "clf__model__nesterov": [False, True],
                "clf__epochs": [50, 100],         # wrapper fit params
                "clf__batch_size": [128, 256],    # wrapper fit params
            }

        cv = self.cv if self.cv is not None else StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=self.param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
            verbose=self.verbose,
            return_train_score=False,
        )
        return grid

    # --- SVC-like API ---
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.grid_ = self._make_grid()
        self.grid_.fit(X, y)
        self.best_estimator_ = self.grid_.best_estimator_
        self.best_params_ = self.grid_.best_params_
        self.best_score_ = self.grid_.best_score_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.best_estimator_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.best_estimator_.score(X, y)

    def save(self, model_path: str, scaler_path: Optional[str] = None):
        """Save the whole fitted wrapper; optionally save just the scaler step too."""
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
