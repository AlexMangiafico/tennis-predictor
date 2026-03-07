"""Train and use the match outcome predictor."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DATA_DIR = Path(__file__).parent.parent / "data"


def _model_path(tour: str) -> Path:
    return DATA_DIR / f"model_{tour}.pkl"


def train(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6)
    model.fit(X, y)
    return model


def save(model: HistGradientBoostingClassifier, feature_cols: list[str], tour: str = "atp") -> None:
    DATA_DIR.mkdir(exist_ok=True)
    path = _model_path(tour)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"Model saved to {path}")


def load(tour: str = "atp") -> tuple[HistGradientBoostingClassifier, list[str]]:
    with open(_model_path(tour), "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_cols"]


def win_probability(model: HistGradientBoostingClassifier, X: pd.DataFrame) -> float:
    """Return probability [0, 1] that the player wins."""
    return float(model.predict_proba(X)[0][1])
