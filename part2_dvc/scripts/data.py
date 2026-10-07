import os

import pandas as pd
import yaml

from configparser import ConfigParser
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


# 1. Подключение к PostgreSQL
def create_connection():

    config = ConfigParser(interpolation=None)
    config.read(".env")

    db = config["default"]

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=db["DB_DESTINATION_USER"],
        password=db["DB_DESTINATION_PASSWORD"],
        host=db["DB_DESTINATION_HOST"],
        port=int(db["DB_DESTINATION_PORT"]),
        database=db["DB_DESTINATION_NAME"],
    )

    conn = create_engine(
        url,
        connect_args={"sslmode": "require"},
    )

    return conn


# 2. Получение исходных данных
def get_data():

    # Чтение параметров
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    # Подключение и выгрузка
    conn = create_connection()

    try:
        data = pd.read_sql(
            "SELECT * FROM public.flats_dataset_clean",
            conn,
            index_col=params["index_col"],
        )
    finally:
        conn.dispose()

    # Сохранение результата
    os.makedirs("data", exist_ok=True)

    data.to_csv(
        "data/initial_data.csv",
        index=True,
        index_label=params["index_col"],
    )


if __name__ == "__main__":
    get_data()
