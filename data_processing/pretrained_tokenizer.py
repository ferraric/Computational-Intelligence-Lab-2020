from typing import List, Tuple

import torch
from data_processing.tokenizer import Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizerFast


class PreTrainedTokenizer(Tokenizer):
    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, max_token_length: int
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def tokenize_tweets(self, tweets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_input = self.tokenizer.batch_encode_plus(
            tweets,
            max_length=self.max_token_length,
            pad_to_max_length=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return tokenized_input["input_ids"], tokenized_input["attention_mask"]
