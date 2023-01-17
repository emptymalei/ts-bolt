import pickle

import pytest
import torch
from gluonts.dataset.common import load_datasets
from gluonts.torch.batchify import batchify

from ts_bolt.datamodules.gluonts import (
    GluonTSDataLoaderConfig,
    GluonTSDataModule,
    GluonTSDataset,
)


@pytest.fixture
def gluonts_dataset(integration_test_dir):

    path = integration_test_dir / "datamodules" / "dataset" / "constant"

    return load_datasets(metadata=path, train=path / "train", test=path / "test")


@pytest.fixture
def gluonts_dataloader_config():
    return GluonTSDataLoaderConfig(**{"batch_size": 2, "transform": None, "collate_fn": batchify})


def test_gluonts_dataset(gluonts_dataset, datamodules_artefacts_dir):

    is_regenerate_artefact = False

    expected_dataset_path = datamodules_artefacts_dir / "gluonts_dataset_to_torch_dataset_expected.pkl"

    g_ds_train = GluonTSDataset(gluonts_dataset=gluonts_dataset, is_train=True)

    if is_regenerate_artefact:
        with open(expected_dataset_path, "wb+") as fp:
            pickle.dump(g_ds_train, fp)

    with open(expected_dataset_path, "rb") as fp:
        g_ds_train_expected = pickle.load(fp)

    assert len(g_ds_train) == len(g_ds_train_expected)


def test_gluonts_datamodule(gluonts_dataset, gluonts_dataloader_config, datamodules_artefacts_dir):

    is_regenerate_artefact = False

    expected_dataloader_path = datamodules_artefacts_dir / "gluonts_dataset_to_torch_dataloader_expected.pkl"

    dm = GluonTSDataModule(
        gluonts_dataset=gluonts_dataset,
        train_dataloader_config=gluonts_dataloader_config,
        test_dataloader_config=gluonts_dataloader_config,
    )

    if is_regenerate_artefact:
        with open(expected_dataloader_path, "wb+") as fp:
            pickle.dump(dm, fp)

    with open(expected_dataloader_path, "rb") as fp:
        dm_expected = pickle.load(fp)

    assert len(dm.train_dataloader()) == len(dm_expected.train_dataloader())

    dm_train_dataloader_values = list(dm.train_dataloader())
    dm_expected_train_dataloader_values = list(dm_expected.train_dataloader())

    for i, j in zip(dm_train_dataloader_values, dm_expected_train_dataloader_values):
        for k in ("target", "feat_static_cat", "feat_static_real"):
            torch.testing.assert_close(i[k], j[k])
