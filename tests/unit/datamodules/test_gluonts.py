import pytest
from gluonts.torch.batchify import batchify

from ts_bolt.datamodules.gluonts import GluonTSDataLoaderConfig, GluonTSDataset


@pytest.fixture
def identity_function(x):
    return x


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {"batch_size": 10, "transform": None, "collate_fn": None},
            {"batch_size": 10, "transform": None, "collate_fn": batchify},
        ),
        (
            {"batch_size": 10, "transform": None, "collate_fn": batchify},
            {"batch_size": 10, "transform": None, "collate_fn": batchify},
        ),
        (
            {"batch_size": 10, "transform": None, "collate_fn": identity_function},
            {"batch_size": 10, "transform": None, "collate_fn": identity_function},
        ),
        (
            {"batch_size": 10, "transform": identity_function, "collate_fn": None},
            {"batch_size": 10, "transform": identity_function, "collate_fn": batchify},
        ),
    ],
    ids=["none_collate_fn", "batchify_collate_fn", "identity_collate_fn", "identity_transform"],
)
def test_gluonts_dataloader_config(params, expected):

    gluonts_dl_config = GluonTSDataLoaderConfig(**params)

    assert gluonts_dl_config.batch_size == expected["batch_size"]
    assert gluonts_dl_config.transform == expected["transform"]
    assert gluonts_dl_config.collate_fn == expected["collate_fn"]
