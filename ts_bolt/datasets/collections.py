from dataclasses import dataclass

from pydantic import BaseModel, HttpUrl
from strenum import StrEnum


class NamedDatasets(StrEnum):
    lai_exchange_rate = "lai_exchange_rate"
    lai_electricity = "lai_electricity"
    lai_solar_al = "lai_solar_al"
    lai_traffic = "lai_traffic"


class DatasetBase(BaseModel):
    name: str
    remote: HttpUrl
    documentation: str
    file_name: str


collections = {
    NamedDatasets.lai_exchange_rate: DatasetBase(
        name=NamedDatasets.lai_exchange_rate,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_exchange_rate}.txt.gz",
    ),
    NamedDatasets.lai_electricity: DatasetBase(
        name=NamedDatasets.lai_electricity,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name="{NamedDatasets.lai_electricity}.txt.gz",
    ),
    NamedDatasets.lai_solar_al: DatasetBase(
        name=NamedDatasets.lai_solar_al,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/solar-energy/solar_AL.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_solar_al}.txt.gz",
    ),
    NamedDatasets.lai_traffic: DatasetBase(
        name=NamedDatasets.lai_traffic,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/traffic/traffic.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_traffic}.txt.gz",
    ),
}
