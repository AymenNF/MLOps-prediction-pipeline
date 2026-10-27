from mlops_pipeline.data import load_data, split
from mlops_pipeline.train import build_model, evaluate


def test_model_beats_quality_gate():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=42)
    model = build_model(C=1.0, max_iter=1000).fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert metrics["accuracy"] > 0.9
    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
