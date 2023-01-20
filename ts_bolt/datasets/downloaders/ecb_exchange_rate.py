from pathlib import Path

import pandas as pd
from loguru import logger


def download_ecb_exchange_rate(target: Path) -> None:
    """Download the ECB Exchange Rate dataset and explore some basic features of the dataset.
    - Dataset Website: https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html
    - Dataset Download Link: https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip
    """
    if target.exists():
        logger.warning(f"dataset exists in {target}, download skipped")
    else:
        df = pd.read_csv(
            "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip",
            compression="zip",
        )

        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        # There are some empty columns.
        columns = [col for col in df.columns if not df[col].isna().all()]

        df = df[columns]

        df.to_csv(target, index=False)
