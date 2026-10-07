import pandas as pd
import pendulum

from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    MetaData,
    Table,
    UniqueConstraint,
)

from steps.cleaning import clean_data


DESTINATION_CONN_ID = "destination_db"
SOURCE_TABLE = "flats_dataset"
TARGET_TABLE = "flats_dataset_clean"


@dag(
    schedule="@once",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ETL"],
)
def clean_flats_dataset():

    @task
    def create_table():
        hook = PostgresHook(DESTINATION_CONN_ID)
        engine = hook.get_sqlalchemy_engine()

        metadata = MetaData()

        Table(
            TARGET_TABLE,
            metadata,
            Column("flat_id", BigInteger),
            Column("building_id", BigInteger),
            Column("floor", BigInteger),
            Column("is_apartment", Boolean),
            Column("kitchen_area", Float),
            Column("living_area", Float),
            Column("rooms", BigInteger),
            Column("studio", Boolean),
            Column("total_area", Float),
            Column("price", BigInteger),
            Column("build_year", BigInteger),
            Column("building_type_int", BigInteger),
            Column("latitude", Float),
            Column("longitude", Float),
            Column("ceiling_height", Float),
            Column("flats_count", BigInteger),
            Column("floors_total", BigInteger),
            Column("has_elevator", Boolean),
            UniqueConstraint("flat_id"),
        )

        metadata.create_all(engine)

    @task
    def extract():
        hook = PostgresHook(DESTINATION_CONN_ID)
        conn = hook.get_conn()

        sql = f"""
        SELECT *
        FROM public.{SOURCE_TABLE}
        """

        data = pd.read_sql(sql, conn)
        conn.close()

        return data

    @task
    def transform(data):
        cleaned_data = clean_data(data)

        return cleaned_data

    @task
    def load(data):
        hook = PostgresHook(DESTINATION_CONN_ID)

        hook.insert_rows(
            table=TARGET_TABLE,
            rows=data.values.tolist(),
            target_fields=data.columns.tolist(),
            replace=True,
            replace_index=["flat_id"],
            commit_every=1000,
        )

    created = create_table()

    data = extract()
    created >> data

    cleaned_data = transform(data)

    load(cleaned_data)


clean_flats_dataset()
