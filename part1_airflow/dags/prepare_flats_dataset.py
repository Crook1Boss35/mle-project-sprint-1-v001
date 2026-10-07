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


DESTINATION_CONN_ID = "destination_db"
TARGET_TABLE = "flats_dataset"


@dag(
    schedule="@once",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ETL"],
)
def prepare_flats_dataset():

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

        sql = """
        SELECT
            f.id AS flat_id,
            f.building_id,
            f.floor,
            f.is_apartment,
            f.kitchen_area,
            f.living_area,
            f.rooms,
            f.studio,
            f.total_area,
            f.price,
            b.build_year,
            b.building_type_int,
            b.latitude,
            b.longitude,
            b.ceiling_height,
            b.flats_count,
            b.floors_total,
            b.has_elevator
        FROM public.flats AS f
        LEFT JOIN public.buildings AS b
            ON f.building_id = b.id
        """

        data = pd.read_sql(sql, conn)
        conn.close()

        return data

    @task
    def transform(data):
        columns = [
            "flat_id",
            "building_id",
            "floor",
            "is_apartment",
            "kitchen_area",
            "living_area",
            "rooms",
            "studio",
            "total_area",
            "price",
            "build_year",
            "building_type_int",
            "latitude",
            "longitude",
            "ceiling_height",
            "flats_count",
            "floors_total",
            "has_elevator",
        ]

        data = data[columns]

        if data["flat_id"].duplicated().any():
            raise ValueError("Duplicate flat_id found after joining tables")

        return data

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

    transformed_data = transform(data)

    load(transformed_data)


prepare_flats_dataset()
