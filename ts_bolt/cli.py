from pathlib import Path

import click
import requests
from loguru import logger

from ts_bolt.datasets.collections import collections as dataset_collections


@click.command()
@click.option(
    "--name",
    type=click.Choice(dataset_collections.keys()),
    help="name of dataset to be downloaded",
    required=True,
)
@click.option(
    "--target",
    type=click.Path(),
    callback=lambda _, __, value: Path(value),
    help="where to save the dataset",
    required=True,
)
def download(name, target):

    dataset = dataset_collections[name]
    if not target.exists():
        target.mkdir(parents=True, exist_ok=False)
    local_file = target / dataset.file_name

    if local_file.exists():
        logger.warning(f"file already exists in {local_file}")
    else:
        r = requests.get(dataset.remote)
        if r.status_code != 200:
            logger.error(f"Can not download {dataset}")
        else:
            with open(local_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)


if __name__ == "__main__":
    download()
