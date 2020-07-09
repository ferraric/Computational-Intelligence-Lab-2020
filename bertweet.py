import os
from argparse import Namespace
from typing import List

import pytorch_lightning as pl
import torch
from bert_sentiment_classifier import BertSentimentClassifier
from bunch import Bunch
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from transformers import RobertaConfig, RobertaModel
from transformers.modeling_roberta import RobertaClassificationHead
from utilities.data_loading import (
    generate_bootstrap_dataset,
    remove_indices_from_test_tweets,
)


class BERTweet(BertSentimentClassifier):
    def __init__(self, config: Bunch) -> None:
        pl.LightningModule.__init__(self)
        self.config = config

        bpe_codes_path = os.path.join(
            config.pretrained_model_base_path, "BERTweet_base_transformers/bpe.codes",
        )
        self.bpe = fastBPE(Namespace(bpe_codes=bpe_codes_path))
        vocab = Dictionary()
        vocab.add_from_file(
            os.path.join(
                config.pretrained_model_base_path,
                "BERTweet_base_transformers/dict.txt",
            )
        )
        self.vocab = vocab

        model_config = RobertaConfig.from_pretrained(
            os.path.join(
                config.pretrained_model_base_path,
                "BERTweet_base_transformers/config.json",
            )
        )
        self.bertweet = RobertaModel.from_pretrained(
            os.path.join(
                config.pretrained_model_base_path,
                "BERTweet_base_transformers/model.bin",
            ),
            config=model_config,
        )
        self.classifier = RobertaClassificationHead(model_config)
        self.loss = CrossEntropyLoss()

    def _load_tweets(self, path: str) -> List[str]:
        def _replace_special_tokens(tweet: str) -> str:
            return tweet.replace("<url>", "HTTPURL").replace("<user>", "@USER")

        return [_replace_special_tokens(tweet) for tweet in super()._load_tweets(path)]

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

    def _pad(self, token_ids: List[List[int]], max_token_length: int) -> torch.Tensor:
        pad_token_id = self.vocab.pad()
        actual_max_token_length = max(map(len, token_ids))
        assert actual_max_token_length <= max_token_length, (
            "max token length set too small, needs to be at least "
            + str(actual_max_token_length)
        )
        return torch.tensor(
            [
                token_ids_per_tweet
                + [pad_token_id] * (max_token_length - len(token_ids_per_tweet))
                for token_ids_per_tweet in token_ids
            ]
        )

    def _generate_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = self.vocab.pad()
        return (token_ids != pad_token_id).float()

    def prepare_data(self) -> None:
        negative_tweets = self._load_unique_tweets(self.config.negative_tweets_path)
        positive_tweets = self._load_unique_tweets(self.config.positive_tweets_path)
        all_tweets = negative_tweets + positive_tweets
        labels = self._generate_labels(len(negative_tweets), len(positive_tweets))

        token_id_list = [
            self._encode(self._split_into_tokens(tweet)) for tweet in all_tweets
        ]
        token_ids = self._pad(token_id_list, self.config.max_tokens_per_tweet)
        attention_mask = self._generate_attention_mask(token_ids)
        self.train_data, self.validation_data = self._train_validation_split(
            self.config.validation_size,
            TensorDataset(token_ids, attention_mask, labels),
        )

        if self.config.do_bootstrap_sampling:
            self.train_data = generate_bootstrap_dataset(self.train_data)

        test_tweets = self._load_tweets(self.config.test_tweets_path)
        test_tweets_index_removed = remove_indices_from_test_tweets(test_tweets)
        test_token_id_list = [
            self._encode(self._split_into_tokens(tweet))
            for tweet in test_tweets_index_removed
        ]
        test_token_ids = self._pad(test_token_id_list, self.config.max_tokens_per_tweet)
        test_attention_mask = self._generate_attention_mask(test_token_ids)
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bertweet(token_ids, attention_mask)
        sequence_output = outputs[0]
        return self.classifier(sequence_output)
