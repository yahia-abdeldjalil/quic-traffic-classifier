# quic-backend/app/calibration.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class PreFitCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """
    Drop-in replacement for:
        CalibratedClassifierCV(estimator, method='sigmoid', cv='prefit')

    Key guarantee: the base estimator is NEVER retrained.
    Only the sigmoid (Platt-scaling) calibration layer is fitted on
    the supplied calibration set — exactly what cv='prefit' did.

    Supports binary and multi-class (OvR) classification.
    """
    def __init__(self, estimator, method='sigmoid'):
        self.estimator = estimator
        self.method = method          # kept for API parity; only 'sigmoid' here

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)

        # Raw probabilities from the *already-fitted* base estimator
        proba = self.estimator.predict_proba(X)   # shape: (n_samples, n_classes)

        if n_cls == 2:
            # ── Binary: Platt-scale on the positive-class column only ──────
            self._calibrators_ = [
                LogisticRegression(solver='lbfgs', max_iter=1_000)
                    .fit(proba[:, [1]], y)
            ]
            self._binary_ = True
        else:
            # ── Multi-class: one-vs-rest, one calibrator per class ──────────
            self._calibrators_ = [
                LogisticRegression(solver='lbfgs', max_iter=1_000)
                    .fit(proba[:, [k]], (y == cls).astype(int))
                for k, cls in enumerate(self.classes_)
            ]
            self._binary_ = False

        return self

    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)

        if self._binary_:
            p1 = self._calibrators_[0].predict_proba(proba[:, [1]])[:, 1]
            return np.column_stack([1.0 - p1, p1])

        # Multi-class: gather calibrated OvR scores, then row-normalize
        raw = np.column_stack([
            cal.predict_proba(proba[:, [k]])[:, 1]
            for k, cal in enumerate(self._calibrators_)
        ])
        return raw / raw.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def n_features_in_(self):           # delegate to base estimator
        return self.estimator.n_features_in_