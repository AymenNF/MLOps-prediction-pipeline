import joblib
import pytest
from fastapi.testclient import TestClient

from mlops_pipeline import serve
from mlops_pipeline.data import load_data
from mlops_pipeline.train import build_model


@pytest.fixture
def client(tmp_path, monkeypatch):
    X, y = load_data()
    path = tmp_path / "model.joblib"
    joblib.dump(build_model(C=1.0, max_iter=1000).fit(X, y), path)
    monkeypatch.setattr(serve, "MODEL_PATH", path)
    with TestClient(serve.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict(client):
    X, _ = load_data()
    resp = client.post("/predict", json={"instances": X.head(3).to_dict(orient="records")})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 3
    assert all(0 <= p <= 1 for p in body["probabilities"])


def test_predict_missing_features(client):
    resp = client.post("/predict", json={"instances": [{"mean radius": 1.0}]})
    assert resp.status_code == 422
