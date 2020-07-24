from typing import List, Tuple

import torch
from data_processing.tokenizer import Tokenizer
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE


class BertweetTokenizer(Tokenizer):
    def __init__(self, max_token_length: int, bpe: fastBPE, vocab: Dictionary):
        super().__init__(max_token_length)
        self.bpe = bpe
        self.vocab = vocab

    def _split_into_tokens(self, tweet: str) -> str:
        return "<s> " + self.bpe.encode(tweet) + " <s>"

    def _encode(self, token_string: str) -> List[int]:
        return (
            self.vocab.encode_line(
                token_string, append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )

    def _pad(self, token_ids: List[List[int]]) -> torch.Tensor:
        pad_token_id = self.vocab.pad()
        actual_max_token_length = max(map(len, token_ids))
        assert actual_max_token_length <= self.max_token_length, (
            "max token length set too small, needs to be at least "
            + str(actual_max_token_length)
        )
        return torch.tensor(
            [
                token_ids_per_tweet
                + [pad_token_id] * (self.max_token_length - len(token_ids_per_tweet))
                for token_ids_per_tweet in token_ids
            ]
        )

    def _generate_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = self.vocab.pad()
        return torch.tensor(token_ids != pad_token_id, dtype=torch.int64)

    def tokenize_tweets(self, tweets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        token_id_list = [
            self._encode(self._split_into_tokens(tweet)) for tweet in tweets
        ]
        token_ids = self._pad(token_id_list)
        attention_mask = self._generate_attention_mask(token_ids)
        return token_ids, attention_mask
