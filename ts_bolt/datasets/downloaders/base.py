from abc import ABC, abstractmethod
from typing import Any


class BaseDownloader(ABC):
    """Base class for a data file downloader."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass
