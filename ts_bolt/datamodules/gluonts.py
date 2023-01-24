from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional

import pytorch_lightning as pl
from gluonts.dataset import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.torch.batchify import batchify
from gluonts.transform import (
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    Transformation,
)
from torch.utils.data import DataLoader, Dataset, IterableDataset


class GluonTSTransformsDefault(Transformation):
    """Default transforms of a gluonts dataset

    ```python
    gluonts_transform = GluonTSTransformsDefault(
        context_length=10,
        prediction_length=5,
    )
    ```

    :param context_length: the length of history input
    :param prediction_length: the length to be forecasted
    """

    def __init__(self, context_length: int, prediction_length: int):
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __call__(self, data_it: Iterable[Dict[str, Any]], is_train: bool):
        mask_unobserved = AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

        training_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ExpectedNumInstanceSampler(
                num_instances=1,
                min_future=self.prediction_length,
            ),
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )

        transforms = mask_unobserved + training_splitter

        return transforms(data_it=data_it, is_train=is_train)


class GluonTSDataset(IterableDataset):
    """An iter style dataset built from a gluonts dataset

    ```python
    from gluonts.dataset.repository.datasets import get_dataset

    gluonts_datasets = get_dataset("electricity")

    dataset = GluonTSDataset(
        dataset = gluonts_datasets.train,
        is_train = True
    )
    ```

    :param dataset: gluonts dataset, e.g., TrainDatasets
    :param is_train: whether the dataset is for training
    :param transform: transformations on dataset, e.g., gluonts.transform.InstanceSplitter
    """

    def __init__(
        self,
        dataset: Dataset,
        is_train: bool,
        transform: Optional[Callable] = None,
        metadata: Optional[Dict[Any, Any]] = None,
    ):
        self.metadata = metadata
        self.dataset = dataset
        self.is_train = is_train

        self.transform = transform
        self.transformed_dataset = self._transform_dataset()

    def __iter__(self):
        for d in self.transformed_dataset:
            yield d

    def _transform_dataset(self) -> List[Dict[str, Any]]:

        if self.transform:
            dataset = self.transform(self.dataset, is_train=self.is_train)
        else:
            dataset = self.dataset

        return dataset


@dataclass
class GluonTSDataLoaderConfig:
    """Configs for dataloaders from a gluonts dataset


    ```python
    dl_config = GluonTSDataLoaderConfig(
        batch_size=2,
        transform=None,
        collate_fn=None,
    )
    ```

    :param batch_size: batch size for the PyTorch DataLoader
    :param transform: transforms of the PyTorch DataLoader, e.g., GluonTSTransformsDefault.
    :param collate_fn: collate_fn of the PyTorch DataLoader, e.g., gluonts.torch.batchify.batchify
    """

    batch_size: int
    transform: Optional[Callable]
    collate_fn: Optional[Callable]

    def __post_init__(self):
        if self.collate_fn is None:
            self.collate_fn = batchify


class GluonTSDataModule(pl.LightningDataModule):
    """LightningDataModule from a gluonts dataset.


    ```python
    from gluonts.dataset.repository.datasets import get_dataset

    gluonts_datasets = get_dataset("electricity")

    train_dl_config = GluonTSDataLoaderConfig(
        batch_size=2,
        transform=None,
        collate_fn=None,
    )
    test_dl_config = GluonTSDataLoaderConfig(
        batch_size=10,
        transform=None,
        collate_fn=None,
    )

    dm = GluonTSDataModule(
        train_dataset = gluonts_datasets.train,
        test_dataset = gluonts_datasets.test,
        train_dataloader_config = train_dl_config,
        test_dataloader_config = test_dl_config,
    )
    ```

    :param train_dataset: gluonts Dataset for training
    :param train_dataset: gluonts Dataset for testing
    :param train_dataloader_config: config for train DataLoader
    :param test_dataloader_config: config for the test DataLoader
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        train_dataloader_config: GluonTSDataLoaderConfig,
        test_dataloader_config: GluonTSDataLoaderConfig,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader_config = train_dataloader_config
        self.test_dataloader_config = test_dataloader_config

    def train_dataloader(self):
        return DataLoader(
            dataset=GluonTSDataset(
                dataset=self.train_dataset,
                is_train=True,
                transform=self.train_dataloader_config.transform,
            ),
            batch_size=self.train_dataloader_config.batch_size,
            collate_fn=self.train_dataloader_config.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=GluonTSDataset(
                dataset=self.test_dataset,
                is_train=False,
                transform=self.test_dataloader_config.transform,
            ),
            batch_size=self.test_dataloader_config.batch_size,
            collate_fn=self.test_dataloader_config.collate_fn,
        )
