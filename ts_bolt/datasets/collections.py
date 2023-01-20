from typing import Callable

from pydantic import BaseModel, HttpUrl
from strenum import StrEnum

from ts_bolt.datasets.downloaders.ecb_exchange_rate import download_ecb_exchange_rate


class NamedDatasets(StrEnum):
    lai_exchange_rate = "lai_exchange_rate"
    lai_electricity = "lai_electricity"
    lai_solar_al = "lai_solar_al"
    lai_traffic = "lai_traffic"
    ecb_exchange_rate = "ecb_exchange_rate"


class RawFileDataset(BaseModel):
    name: str
    remote: HttpUrl
    documentation: str
    file_name: str


class DownloaderDataset(BaseModel):
    name: str
    downloader: Callable
    documentation: str
    file_name: str


collections = {
    NamedDatasets.lai_exchange_rate: RawFileDataset(
        name=NamedDatasets.lai_exchange_rate,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_exchange_rate}.txt.gz",
    ),
    NamedDatasets.lai_electricity: RawFileDataset(
        name=NamedDatasets.lai_electricity,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name="{NamedDatasets.lai_electricity}.txt.gz",
    ),
    NamedDatasets.lai_solar_al: RawFileDataset(
        name=NamedDatasets.lai_solar_al,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/solar-energy/solar_AL.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_solar_al}.txt.gz",
    ),
    NamedDatasets.lai_traffic: RawFileDataset(
        name=NamedDatasets.lai_traffic,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/traffic/traffic.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_traffic}.txt.gz",
    ),
    NamedDatasets.ecb_exchange_rate: DownloaderDataset(
        name=NamedDatasets.ecb_exchange_rate,
        downloader=download_ecb_exchange_rate,
        documentation="https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html",
        file_name=f"{NamedDatasets.ecb_exchange_rate}.csv",
    ),
}
