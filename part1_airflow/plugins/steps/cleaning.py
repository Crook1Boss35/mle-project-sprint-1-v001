import numpy as np
import pandas as pd


def fix_ceiling_height(value):
    if pd.isna(value):
        return value

    if 10 <= value < 100:
        corrected = value / 10
        if 2 <= corrected <= 8:
            return corrected

    return value


def clean_data(data, price_iqr_factor=3.0):
    data = data.copy()

    # 1. Явные и неявные дубликаты
    data = data.drop_duplicates()

    implicit_duplicate_cols = [
        col for col in data.columns
        if col != "flat_id"
    ]

    data = data.drop_duplicates(
        subset=implicit_duplicate_cols,
        keep="first",
    )

    # 2. Пропуски
    data = data.dropna().copy()

    # 3. Логические ограничения
    current_year = pd.Timestamp.now().year

    logical_error_mask = (
        (data["price"] <= 0)
        | (data["total_area"] <= 0)
        | (data["kitchen_area"] < 0)
        | (data["living_area"] < 0)
        | (data["rooms"] <= 0)
        | (data["floor"] <= 0)
        | (data["floors_total"] <= 0)
        | (data["floor"] > data["floors_total"])
        | (data["kitchen_area"] > data["total_area"])
        | (data["living_area"] > data["total_area"])
        | (
            data["kitchen_area"]
            + data["living_area"]
            > data["total_area"]
        )
        | (data["build_year"] < 1800)
        | (data["build_year"] > current_year)
        | (data["ceiling_height"] <= 0)
        | (~data["latitude"].between(-90, 90))
        | (~data["longitude"].between(-180, 180))
    )

    data = data.loc[~logical_error_mask].copy()

    # 4. Высота потолков
    data["ceiling_height"] = (
        data["ceiling_height"]
        .apply(fix_ceiling_height)
    )

    data = data.loc[
        data["ceiling_height"].between(2, 8)
    ].copy()

    # 5. Экстремальные значения цены через цену за м²
    price_per_sqm = data["price"] / data["total_area"]
    log_price_per_sqm = np.log1p(price_per_sqm)

    q1 = log_price_per_sqm.quantile(0.25)
    q3 = log_price_per_sqm.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - price_iqr_factor * iqr
    upper_bound = q3 + price_iqr_factor * iqr

    price_valid_mask = log_price_per_sqm.between(
        lower_bound,
        upper_bound,
    )

    data = data.loc[price_valid_mask].copy()

    return data.reset_index(drop=True)
