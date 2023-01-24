import pickle

import numpy as np
import pytest
import torch
from gluonts.dataset.common import load_datasets
from gluonts.torch.batchify import batchify

from ts_bolt.datamodules.gluonts import (
    GluonTSDataLoaderConfig,
    GluonTSDataModule,
    GluonTSDataset,
    GluonTSTransformsDefault,
)


@pytest.fixture
def context_length():
    return 10


@pytest.fixture
def prediction_length():
    return 3


@pytest.fixture
def gluonts_datasets(integration_test_dir):

    path = integration_test_dir / "datamodules" / "dataset" / "constant"

    return load_datasets(metadata=path, train=path / "train", test=path / "test")


@pytest.fixture
def gluonts_dataloader_config():
    return GluonTSDataLoaderConfig(
        **{"batch_size": 2, "transform": None, "collate_fn": batchify}
    )


@pytest.fixture
def gluonts_transform(context_length, prediction_length):
    return GluonTSTransformsDefault(
        context_length=context_length, prediction_length=prediction_length
    )


def test_gluonts_datasets(gluonts_datasets, datamodules_artefacts_dir):

    is_regenerate_artefact = False

    expected_dataset_path = (
        datamodules_artefacts_dir / "gluonts_dataset_to_torch_dataset_expected.pkl"
    )

    g_ds_train = GluonTSDataset(dataset=gluonts_datasets.train, is_train=True)

    if is_regenerate_artefact:
        with open(expected_dataset_path, "wb+") as fp:
            pickle.dump(list(g_ds_train), fp)

    with open(expected_dataset_path, "rb") as fp:
        g_ds_train_expected = pickle.load(fp)

    assert len(list(g_ds_train)) == len(g_ds_train_expected)

    for i in range(len(g_ds_train_expected)):
        for k in g_ds_train_expected[i]:
            g_ds_train_expected[i][k] == list(g_ds_train)[i][k]


def test_gluonts_datasets_with_transform(
    gluonts_datasets, gluonts_transform, datamodules_artefacts_dir
):

    is_regenerate_artefact = False

    expected_dataset_path = (
        datamodules_artefacts_dir
        / "gluonts_dataset_with_transform_to_torch_dataset_expected.pkl"
    )

    np.random.seed(42)

    g_ds_train = GluonTSDataset(
        dataset=gluonts_datasets.train, is_train=True, transform=gluonts_transform
    )
    g_ds_train_values = list(g_ds_train)

    if is_regenerate_artefact:
        with open(expected_dataset_path, "wb+") as fp:
            pickle.dump(g_ds_train_values, fp)

    with open(expected_dataset_path, "rb") as fp:
        g_ds_train_expected = pickle.load(fp)

    assert len(g_ds_train_values) == len(g_ds_train_expected)

    for i in range(len(g_ds_train_expected)):
        for k in g_ds_train_expected[i]:
            g_ds_train_expected[i][k] == g_ds_train_values[i][k]


def test_gluonts_datamodule(
    gluonts_datasets, gluonts_dataloader_config, datamodules_artefacts_dir
):

    is_regenerate_artefact = False

    expected_train_dataloader_path = (
        datamodules_artefacts_dir
        / "gluonts_dataset_to_torch_train_dataloader_expected.pkl"
    )
    expected_test_dataloader_path = (
        datamodules_artefacts_dir
        / "gluonts_dataset_to_torch_test_dataloader_expected.pkl"
    )

    dm = GluonTSDataModule(
        train_dataset=gluonts_datasets.train,
        test_dataset=gluonts_datasets.test,
        train_dataloader_config=gluonts_dataloader_config,
        test_dataloader_config=gluonts_dataloader_config,
    )

    if is_regenerate_artefact:
        with open(expected_train_dataloader_path, "wb+") as fp:
            pickle.dump(list(dm.train_dataloader()), fp)

        with open(expected_test_dataloader_path, "wb+") as fp:
            pickle.dump(list(dm.test_dataloader()), fp)

    with open(expected_train_dataloader_path, "rb") as fp:
        dm_train_dataloader_expected = pickle.load(fp)

    with open(expected_test_dataloader_path, "rb") as fp:
        dm_test_dataloader_expected = pickle.load(fp)

    assert len(list(dm.train_dataloader())) == len(list(dm_train_dataloader_expected))
    assert len(list(dm.test_dataloader())) == len(list(dm_test_dataloader_expected))

    dm_train_dataloader_values = list(dm.train_dataloader())

    for i, j in zip(dm_train_dataloader_values, dm_train_dataloader_expected):
        for k in ("target", "feat_static_cat", "feat_static_real"):
            torch.testing.assert_close(i[k], j[k])
