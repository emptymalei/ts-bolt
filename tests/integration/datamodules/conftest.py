import pytest


@pytest.fixture(scope="session")
def datamodules_artefacts_dir(integration_test_dir):
    return integration_test_dir / "datamodules" / "artefacts"
