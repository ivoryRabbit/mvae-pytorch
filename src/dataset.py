import math
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pandas import DataFrame
from torch.utils import data


class Transform(ABC):
    @abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass


class DenseIndexing(Transform):
    def __init__(self, item_map: DataFrame):
        self.item_map = self.refine_item_map(item_map)
        self.indexer = np.vectorize(self.item_map.get)

    @staticmethod
    def refine_item_map(item_map: DataFrame) -> Dict[int, int]:
        return {Id: idx for Id, idx in item_map[["item_id", "item_idx"]].values}

    def __call__(self, batch):
        inputs, targets = batch["inputs"], batch["targets"]
        inputs, targets = map(self.indexer, (inputs, targets))
        return {"inputs": inputs, "targets": targets}


class OneHotEncoding(Transform):
    def __init__(self, output_size: int):
        self.output_size = output_size

    def to_one_hot(self, indices) -> np.ndarray:
        one_hot = np.zeros(shape=(self.output_size, ), dtype=np.float32)
        one_hot[indices] = 1.0
        return one_hot

    def __call__(self, batch):
        inputs, targets = batch["inputs"], batch["targets"]
        inputs, targets = map(self.to_one_hot, (inputs, targets))
        return {"inputs": inputs, "targets": targets}


class Corruption(Transform):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, batch):
        inputs, targets = batch["inputs"], batch["targets"]
        size = math.ceil(len(inputs) * 0.9)
        inputs = random.choices(inputs, k=size)
        return {"inputs": inputs, "targets": targets}


class Dataset(data.Dataset):
    def __init__(self, df: DataFrame, transforms: Optional[List[Transform]] = None):
        self.df = self.aggregate_item(df)
        self.transforms = transforms

    @staticmethod
    def aggregate_item(df) -> DataFrame:
        return df.groupby("user_id", as_index=False)[["item_idx"]].agg(list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        items: List[int] = self.df.iloc[idx, 1]
        batch = {"inputs": items, "targets": items}

        if isinstance(self.transforms, list):
            for transform in self.transforms:
                batch = transform(batch)

        return batch
