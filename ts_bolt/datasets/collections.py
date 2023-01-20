from dataclasses import dataclass

from pydantic import BaseModel, HttpUrl
from strenum import StrEnum


class NamedDatasets(StrEnum):
    exchange_rate = "exchange_rate"


class DatasetBase(BaseModel):
    name: str
    remote: HttpUrl
    documentation: str
    file_name: str


collections = {
    NamedDatasets.exchange_rate: DatasetBase(
        name=NamedDatasets.exchange_rate,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name="exchange_rate.txt.gz",
    )
}
