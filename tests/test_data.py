import pytest

from mlops_pipeline.data import FEATURES, load_data, split, validate


def test_load_and_validate():
    X, y = load_data()
    validate(X, y)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) > 0


def test_validate_rejects_nan():
    X, y = load_data()
    X.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate(X, y)


def test_split_is_stratified():
    X, y = load_data()
    _, _, y_train, y_test = split(X, y, test_size=0.2, random_state=0)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
