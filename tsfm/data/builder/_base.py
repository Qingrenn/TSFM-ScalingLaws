import abc
from typing import Any, Callable

from torch.utils.data import ConcatDataset, Dataset

from tsfm.transform import Transformation

import logging
log = logging.getLogger(__name__)

# TODO: Add __repr__
class DatasetBuilder(abc.ABC):
    @abc.abstractmethod
    def build_dataset(self, *args, **kwargs): ...

    @abc.abstractmethod
    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> Dataset: ...


class ConcatDatasetBuilder(DatasetBuilder):
    def __init__(self, *builders: DatasetBuilder):
        super().__init__()
        assert len(builders) > 0, "Must provide at least one builder to ConcatBuilder"
        assert all(
            isinstance(builder, DatasetBuilder) for builder in builders
        ), "All builders must be instances of DatasetBuilder"
        self.builders: tuple[DatasetBuilder, ...] = builders

    def build_dataset(self):
        raise ValueError(
            "Do not use ConcatBuilder to build datasets, build sub datasets individually instead."
        )

    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> ConcatDataset:
        return ConcatDataset(
            [builder.load_dataset(transform_map) for builder in self.builders]
        )


class SafeConcatDatasetBuilder(ConcatDatasetBuilder):
    def __init__(self, *builders: DatasetBuilder):
        super().__init__(*builders)
        self.builders = builders

    def load_dataset(
            self, transform_map: dict[Any, Callable[..., Transformation]]
        ) -> ConcatDataset:
            datasets = []
            for builder in self.builders:
                try:
                    datasets.append(builder.safeload_dataset(transform_map))
                except Exception as e:
                    log.error(f"Error loading dataset from {builder}: {e}")
                    continue
                
            return ConcatDataset(datasets)
    
