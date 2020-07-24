from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class Tokenizer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def tokenize_tweets(self, tweets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
