# CardioMap: builds the frontend, trains the served models, serves everything on :8000.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt
COPY data/ data/
COPY ml/ ml/
COPY artifacts/ artifacts/
COPY backend/ backend/
COPY tests/test_leakage.py tests/test_api.py tests/test_explain.py tests/
# models.joblib is rebuilt here (about 10 s), so the image never depends on a pickled file from another machine.
RUN python ml/train.py && python -m pytest -q tests
COPY --from=web /web/dist frontend/dist
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
