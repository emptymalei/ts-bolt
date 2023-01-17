import pytest


@pytest.fixture
def datamodules_artefacts_dir(integration_test_dir):
    return integration_test_dir / "datamodules" / "artefacts"
