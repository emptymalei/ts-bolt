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
    mbohlkeschneider_electricity_nips = "mbohlkeschneider_electricity_nips"
    mbohlkeschneider_exchange_rate_nips = "mbohlkeschneider_exchange_rate_nips"
    mbohlkeschneider_solar_nips = "mbohlkeschneider_solar_nips"
    mbohlkeschneider_wiki_rolling_nips = "mbohlkeschneider_wiki_rolling_nips"
    mbohlkeschneider_traffic_nips = "mbohlkeschneider_traffic_nips"
    mbohlkeschneider_taxi_30min = "mbohlkeschneider_taxi_30min"


class RawFileDataset(BaseModel):
    name: str
    remote: HttpUrl
    documentation: str
    file_name: str
    description: str


class DownloaderDataset(BaseModel):
    name: str
    downloader: Callable
    documentation: str
    file_name: str
    description: str


collections = {
    NamedDatasets.lai_exchange_rate: RawFileDataset(
        name=NamedDatasets.lai_exchange_rate,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_exchange_rate}.txt.gz",
        description="exchange rates in tabular format",
    ),
    NamedDatasets.lai_electricity: RawFileDataset(
        name=NamedDatasets.lai_electricity,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name="{NamedDatasets.lai_electricity}.txt.gz",
        description="UCI electricity data in tabular format",
    ),
    NamedDatasets.lai_solar_al: RawFileDataset(
        name=NamedDatasets.lai_solar_al,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/solar-energy/solar_AL.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_solar_al}.txt.gz",
        description="solar AL in tabular format",
    ),
    NamedDatasets.lai_traffic: RawFileDataset(
        name=NamedDatasets.lai_traffic,
        remote="https://github.com/laiguokun/multivariate-time-series-data/raw/master/traffic/traffic.txt.gz",
        documentation="https://github.com/laiguokun/multivariate-time-series-data",
        file_name=f"{NamedDatasets.lai_traffic}.txt.gz",
        description="traffic dataset in tabular format",
    ),
    NamedDatasets.ecb_exchange_rate: DownloaderDataset(
        name=NamedDatasets.ecb_exchange_rate,
        downloader=download_ecb_exchange_rate,
        documentation="https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html",
        file_name=f"{NamedDatasets.ecb_exchange_rate}.csv",
        description="ecb exchange rate downloaded live in csv format",
    ),
    NamedDatasets.mbohlkeschneider_electricity_nips: RawFileDataset(
        name=NamedDatasets.mbohlkeschneider_electricity_nips,
        remote="https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/electricity_nips.tar.gz",
        documentation="https://arxiv.org/abs/2101.12072",
        file_name=f"{NamedDatasets.mbohlkeschneider_electricity_nips}.tar.gz",
        description="gluonts dataset format",
    ),
    NamedDatasets.mbohlkeschneider_exchange_rate_nips: RawFileDataset(
        name=NamedDatasets.mbohlkeschneider_exchange_rate_nips,
        remote="https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/exchange_rate_nips.tar.gz",
        documentation="https://arxiv.org/abs/2101.12072",
        file_name=f"{NamedDatasets.mbohlkeschneider_exchange_rate_nips}.tar.gz",
        description="gluonts dataset format",
    ),
    NamedDatasets.mbohlkeschneider_solar_nips: RawFileDataset(
        name=NamedDatasets.mbohlkeschneider_solar_nips,
        remote="https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/solar_nips.tar.gz",
        documentation="https://arxiv.org/abs/2101.12072",
        file_name=f"{NamedDatasets.mbohlkeschneider_solar_nips}.tar.gz",
        description="gluonts dataset format",
    ),
    NamedDatasets.mbohlkeschneider_wiki_rolling_nips: RawFileDataset(
        name=NamedDatasets.mbohlkeschneider_wiki_rolling_nips,
        remote="https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/wiki-rolling_nips.tar.gz",
        documentation="https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release/datasets",
        file_name=f"{NamedDatasets.mbohlkeschneider_wiki_rolling_nips}.tar.gz",
        description="gluonts dataset format",
    ),
    NamedDatasets.mbohlkeschneider_traffic_nips: RawFileDataset(
        name=NamedDatasets.mbohlkeschneider_traffic_nips,
        remote="https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/traffic_nips.tar.gz",
        documentation="https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release/datasets",
        file_name=f"{NamedDatasets.mbohlkeschneider_traffic_nips}.tar.gz",
        description="gluonts dataset format",
    ),
    NamedDatasets.mbohlkeschneider_taxi_30min: RawFileDataset(
        name=NamedDatasets.mbohlkeschneider_taxi_30min,
        remote="https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/taxi_30min.tar.gz",
        documentation="https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release/datasets",
        file_name=f"{NamedDatasets.mbohlkeschneider_taxi_30min}.tar.gz",
        description="gluonts dataset format",
    ),
}
