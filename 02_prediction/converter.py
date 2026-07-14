from typing import Iterable
from pandas import DataFrame


def enumerator(values: Iterable[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


def columns_mapper(columns_str: list[str], df: DataFrame) -> dict[str, dict[str, int]]:
    return {column: enumerator(df[column].unique()) for column in columns_str}


def convert_x(df: DataFrame, mapper: dict[str, dict[str, int]]) -> DataFrame:
    return DataFrame(
        {
            column: df[column].map(mapper[column]) if column in mapper else df[column]
            for column in df.columns
        }
    )
