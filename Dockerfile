FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml params.yaml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . \
    && python -m mlops_pipeline.train

EXPOSE 8000
CMD ["uvicorn", "mlops_pipeline.serve:app", "--host", "0.0.0.0", "--port", "8000"]
