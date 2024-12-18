from functools import reduce
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from ground_slash.program import Constraint, Naf, Number, PredLiteral, SymbolicConstant
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST


class MNISTAddition(Dataset):
    """TODO"""

    def __init__(
        self,
        n: int,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        digits: Optional[Iterable[int]] = None,
        seed: int = None,
    ) -> None:
        """TODO"""
        self.n = n
        self.transform = transform
        self.train = train
        self.root = root

        # get regular MNIST dataset
        self.mnist = MNIST(
            root=root, train=train, transform=transform, download=download
        )

        if digits is not None:
            self.digits = set(digits) if not isinstance(digits, set) else digits
            self.mnist = Subset(
                self.mnist,
                torch.where(
                    reduce(
                        torch.logical_or,
                        [self.mnist.targets == digit for digit in digits],
                    )
                )[0],
            )
        else:
            self.digits = set(range(10))

        if seed is not None:
            torch.manual_seed(seed)

        self.data = []

        for ids in torch.split(torch.randperm(len(self.mnist)), self.n):
            # chunk is not complete
            if len(ids) != self.n:
                continue

            x, y = tuple(zip(*tuple(self.mnist[i] for i in ids)))

            self.data.append(
                (
                    x,
                    sum(y),
                )
            )

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, ...], int]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def to_queries(self, y: Iterable[int]) -> List[Constraint]:
        return [
            Constraint(
                Naf(
                    PredLiteral(
                        "addition",
                        *tuple(SymbolicConstant(f"i{i+1}") for i in range(self.n)),
                        Number(y_i.item()),
                    )
                )
            )
            for y_i in y
        ]
