from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset


class DataFrameDataset(Dataset):
    """A dataset from a pandas dataframe
    :param dataframe: input dataframe with a DatetimeIndex.
    :param contex_length: length of input in time dimension
    :param horizon: future length to be forecasted
    """

    def __init__(self, dataframe: pd.DataFrame, context_length: int, horizon: int):
        super().__init__()
        self.dataframe = dataframe
        self.context_length = context_length
        self.horzion = horizon
        self.dataframe_rows = len(self.dataframe)
        self.length = self.dataframe_rows - self.context_length - self.horzion

    def moving_slicing(self, idx):

        x, y = (
            self.dataframe[idx : self.context_length + idx].values,
            self.dataframe[
                self.context_length + idx : self.context_length + self.horzion + idx
            ].values,
        )
        return x, y

    def _validate_dataframe(self):
        """Validate the input dataframe.
        - We require the dataframe index to be DatetimeIndex.
        - This dataset is null aversion.
        - Dataframe index should be sorted.
        """

        if not isinstance(
            self.dataframe.index, pd.core.indexes.datetimes.DatetimeIndex
        ):
            raise TypeError(
                f"Type of the dataframe index is not DatetimeIndex: {type(self.dataframe.index)}"
            )

        has_na = self.dataframe.isnull().values.any()

        if has_na:
            logger.warning(f"Dataframe has null")

        has_index_sorted = self.dataframe.index.equals(
            self.dataframe.index.sort_values()
        )

        if not has_index_sorted:
            logger.warning(f"Dataframe index is not sorted")

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("End of dataset")
        return self.moving_slicing(idx)

    def __len__(self):
        return self.length
