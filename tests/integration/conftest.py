from pathlib import Path

import pytest


@pytest.fixture
def integration_test_dir():
    return Path(__file__).parent.resolve()
