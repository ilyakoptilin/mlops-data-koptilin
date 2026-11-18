# MLOps: Версионирование данных и управление экспериментами

Репозиторий практик 3 и 4 по дисциплине «Операционализация моделей машинного обучения» (Коптилин Илья).

## Восстановление окружения и данных (Практика 3)

```bash
poetry install
poetry run dvc pull
poetry run python check_data.py --expect iris wine
```

## Запуск экспериментов и тестов (Практика 4)

```bash
# Локальный запуск (SQLite backend)
poetry run python run_experiments.py --tracking-uri sqlite:///mlflow.db --dataset-id 998f9aad4e5245509a421e596a01ebeb --expected-sha256 724d80b05e39c7a72fe7631de59583c764049137f9d7089b01b5460e212c75c0

# Просмотр MLflow UI
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080

# Запуск через MLflow Tracking Server
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 8080
poetry run python run_experiments.py --tracking-uri http://127.0.0.1:8080

# Автотесты структуры runs
poetry run pytest -v
```
