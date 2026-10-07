import json
import os

import joblib
import pandas as pd
import yaml

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


def evaluate_model():

    # 1. Чтение параметров
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    # 2. Загрузка данных
    data = pd.read_csv(
        "data/initial_data.csv",
        index_col=params["index_col"],
    )

    # 3. Загрузка обученного Pipeline
    pipeline = joblib.load(
        "models/fitted_model.pkl"
    )

    # 4. Разделение признаков и целевой переменной
    X = data.drop(
        columns=[params["target_col"]],
    )

    y = data[params["target_col"]]

    # 5. Повторяем тот же train / test split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    # 6. Предсказания
    y_pred = pipeline.predict(X_test)

    # 7. Метрики
    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    rmse = mean_squared_error(
        y_test,
        y_pred,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred,
    )

    results = {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 3),
        "test_size": len(y_test),
    }

    # 8. Сохранение результатов
    os.makedirs("cv_results", exist_ok=True)

    with open(
        "cv_results/cv_res.json",
        "w",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )


if __name__ == "__main__":
    evaluate_model()
