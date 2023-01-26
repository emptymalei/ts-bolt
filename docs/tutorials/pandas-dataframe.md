# Pandas DataFrame as Dataset

In this tutorial, we show a few examples using pandas dataframe as dataset.

## `ts_bolt.datamodules.pandas.DataFrameDataset`

`ts_bolt.datamodules.pandas.DataFrameDataset` takes a pandas dataframe and converts it to a PyTorch dataset.


```python
import pandas as pd

from ts_bolt.datamodules.pandas import DataFrameDataset

dates = pd.date_range("2021-01-01", "2021-04-01", freq="D")

df_a = pd.DataFrame(
    {
        "date": dates,
        "target": range(len(dates)),
        "item_id": ["A"] * len(dates)
    }
)
df_b = pd.DataFrame(
    {
        "date": dates,
        "target": range(len(dates)),
        "item_id": ["B"] * len(dates)
    }
)

df_long = pd.concat([df_a, df_b])

df_wide = (
    df_long
    .pivot(index="date", columns="item_id", values="target")
)
```

With this wide dataframe, we can construct a PyTorch dataset

```python
dfds = DataFrameDataset(dataframe=df_wide, context_length=3, horizon=2)

next(iter(dfds))
```

## Using GluonTS PandasDataset

`ts_bolt.datamodules.gluonts` provides a generic connection between gluonts datasets and pytorch dataloader.


```python
gluonts_pds = PandasDataset.from_long_dataframe(
    pandas_dataframe, target="target", item_id="item_id"
)

ds = GluonTSDataset(dataset=gluonts_pds, is_train=True, transform=gluonts_transform)
```

## DataLoader

Once we obtained the dataset, a dataloader can be constructed the PyTorch way.

```python
dl = DataLoader(ds, batch_size=2, collate_fn=lambda data: data)
next(iter(dl))
```
