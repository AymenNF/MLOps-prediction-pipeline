# MLOps pipeline

A small, end-to-end MLOps starter: a binary classifier (scikit-learn, on the built-in
breast-cancer dataset) that goes through the full lifecycle:

| Stage        | What happens                                              | Where                          |
|--------------|-----------------------------------------------------------|--------------------------------|
| Data         | Load + validate (schema, NaNs, binary target), split      | `src/mlops_pipeline/data.py`   |
| Train        | Scaler + logistic regression, params from `params.yaml`   | `src/mlops_pipeline/train.py`  |
| Track        | Params, metrics and model logged to MLflow                | `mlflow.db`, `mlruns/`         |
| Quality gate | Training fails if test accuracy < `min_accuracy`          | `params.yaml`                  |
| Serve        | FastAPI `/predict`, `/health`, `/features`                | `src/mlops_pipeline/serve.py`  |
| Test         | Data, model and API tests                                 | `tests/`                       |
| Ship         | Docker image + GitHub Actions CI (lint, test, train, build) | `Dockerfile`, `.github/`     |

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .

python -m mlops_pipeline.train            # train + log to MLflow, saves models/model.joblib
pytest                                    # run tests
uvicorn mlops_pipeline.serve:app --reload # API at http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment UI at http://127.0.0.1:5000
```

## Docker

```bash
docker build -t mlops-pipeline .
docker run -p 8000:8000 mlops-pipeline
```

## Next steps

- Swap in your own dataset in `data.py` (and version it, e.g. with DVC)
- Try other models / hyperparameter search, compare runs in MLflow
- Promote models via the MLflow Model Registry instead of `models/model.joblib`
- Add drift monitoring on incoming `/predict` traffic
