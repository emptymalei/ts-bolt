import pandas as pd
import pytest


@pytest.fixture
def datamodules_artefacts_dir(integration_test_dir):
    return integration_test_dir / "datamodules" / "artefacts"


@pytest.fixture
def pandas_dataframe():
    dates = pd.date_range("2021-01-01", "2021-04-01", freq="D")

    df_a = pd.DataFrame(
        {"date": dates, "target": range(len(dates)), "item_id": ["A"] * len(dates)}
    )
    df_b = pd.DataFrame(
        {"date": dates, "target": range(len(dates)), "item_id": ["B"] * len(dates)}
    )

    return pd.concat([df_a, df_b])
