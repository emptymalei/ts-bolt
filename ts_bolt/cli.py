from pathlib import Path

import click
import requests
from loguru import logger

from ts_bolt.datasets.collections import collections as dataset_collections


@click.group(invoke_without_command=True)
@click.pass_context
def bolt(ctx):
    if ctx.invoked_subcommand is None:
        click.echo("Use bolt --help for help.")
    else:
        pass


@bolt.command()
@click.option(
    "--name",
    default=None,
    type=click.Choice(dataset_collections.keys()),
    help="name of dataset to be check",
    required=False,
)
def list(name):

    if name is None:
        click.echo_via_pager("\n".join(f"{d}" for d in dataset_collections))
    else:
        click.secho(f"{name} definition: {dataset_collections[name]}")


@bolt.command()
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
    pass
