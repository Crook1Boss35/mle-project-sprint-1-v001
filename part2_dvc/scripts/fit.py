import os

import joblib
import pandas as pd
import yaml

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def fit_model():

    # 1. Чтение параметров
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    # 2. Загрузка данных
    data = pd.read_csv(
        "data/initial_data.csv",
        index_col=params["index_col"],
    )

    # 3. Признаки и целевая переменная
    X = data.drop(
        columns=[
            params["target_col"],
            *params["drop_cols"],
        ],
        errors="ignore",
    )

    y = data[params["target_col"]]

    # 4. Train / test split
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    # 5. Категориальные признаки задаём явно
    cat_cols = params["categorical_cols"]

    # Остальные признаки числовые
    num_cols = [
        col
        for col in X_train.columns
        if col not in cat_cols
    ]

    # 6. Предобработка
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    drop=params["one_hot_drop"],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ]
    )

    # 7. Модель
    model = CatBoostRegressor(
        iterations=params["iterations"],
        random_seed=params["random_state"],
        verbose=False,
        allow_writing_files=False,
    )

    # 8. Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # 9. Обучение
    pipeline.fit(X_train, y_train)

    # 10. Сохранение модели
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        pipeline,
        "models/fitted_model.pkl",
    )


if __name__ == "__main__":
    fit_model()
