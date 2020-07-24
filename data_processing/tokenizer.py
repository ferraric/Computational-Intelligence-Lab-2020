from typing import List, Tuple

import torch
from transformers.tokenization_utils import PreTrainedTokenizerFast


class Tokenizer:
    def __init__(
        self, max_token_length: int, tokenizer: PreTrainedTokenizerFast = None
    ):
        self.max_token_length = max_token_length
        self.tokenizer = tokenizer

    def tokenize_tweets(self, tweets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is not None:
            tokenized_input = self.tokenizer.batch_encode_plus(
                tweets,
                max_length=self.max_token_length,
                pad_to_max_length=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            return tokenized_input["input_ids"], tokenized_input["attention_mask"]
        else:
            raise ValueError("Tokenizer is None")
