from typing import Iterable
import pandas as pd


def enumerator(values: Iterable[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


def columns_mapper(columns_str: list[str], df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {column: enumerator(df[column].unique()) for column in columns_str}


def convert_x(df: pd.DataFrame, mapper: dict[str, dict[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: df[column].map(mapper[column]) if column in mapper else df[column]
            for column in df.columns
        }
    )
