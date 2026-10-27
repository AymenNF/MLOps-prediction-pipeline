import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

FEATURES: list[str] = list(load_breast_cancer().feature_names)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the bundled breast-cancer dataset (no download needed)."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return X, y


def validate(X: pd.DataFrame, y: pd.Series) -> None:
    """Basic data checks that fail the pipeline early on bad input."""
    missing = set(FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if X.isna().any().any():
        raise ValueError("Feature data contains NaN values")
    if len(X) != len(y):
        raise ValueError("Features and target have different lengths")
    if set(y.unique()) - {0, 1}:
        raise ValueError("Target must be binary (0/1)")


def split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
