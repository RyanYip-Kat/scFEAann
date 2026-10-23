"""Dataset wrapper matching the original scFEA implementation."""

from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self,
        data: Any,
        label: Any,
        info: Any,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.data = data
        self.label = label
        self.info = info
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        value = self.data[index]
        label = self.label[index]
        info = self.info[index]
        if self.transform:
            value = self.transform(value)
        return value, label, info

    def __len__(self) -> int:
        return len(self.data)
