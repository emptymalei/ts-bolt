import pandas as pd

from ts_bolt.datamodules.pandas import DataFrameDataset


def test_dataframe_dataset():

    dates = pd.date_range("2021-01-01", "2021-04-01", freq="D")
    df = pd.DataFrame({"date": dates, "value_1": range(len(dates))})

    dfds = DataFrameDataset(dataframe=df, context_length=7, horizon=2)

    assert len(dfds) == 82

    for d in dfds:
        assert d[0].shape == (7, 2)
        assert d[1].shape == (2, 2)
