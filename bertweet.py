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


class BERTweet(BertSentimentClassifier):
    def __init__(self, config: Bunch) -> None:
        pl.LightningModule.__init__(self)
        self.config = config
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
        def replace_special_tokens(tweet: str) -> str:
            return tweet.replace("<url>", "HTTPURL").replace("<user>", "@USER")

        tweets = super()._load_tweets(path)
        return list(map(replace_special_tokens, tweets))

    def prepare_data(self) -> None:
        bpe_codes_path = os.path.join(
            self.config.pretrained_model_base_path,
            "BERTweet_base_transformers/bpe.codes",
        )
        bpe = fastBPE(Namespace(bpe_codes=bpe_codes_path))
        vocab = Dictionary()
        vocab.add_from_file(
            os.path.join(
                self.config.pretrained_model_base_path,
                "BERTweet_base_transformers/dict.txt",
            )
        )

        def _split_into_tokens(tweet: str) -> str:
            return "<s> " + bpe.encode(tweet) + " <s>"

        def _encode(token_string: str) -> List[int]:
            return (
                vocab.encode_line(
                    token_string, append_eos=False, add_if_not_exist=False
                )
                .long()
                .tolist()
            )

        negative_tweets = self._load_tweets(self.config.negative_tweets_path)
        positive_tweets = self._load_tweets(self.config.positive_tweets_path)
        all_tweets = negative_tweets + positive_tweets
        labels = self._generate_labels(len(negative_tweets), len(positive_tweets))

        token_strings = list(map(_split_into_tokens, all_tweets))
        token_id_list = list(map(_encode, token_strings))

        test_tweets = self._load_tweets(self.config.test_tweets_path)
        test_tweets_index_removed = [
            self._remove_index_from_test_tweet(tweet) for tweet in test_tweets
        ]
        token_strings = list(map(_split_into_tokens, test_tweets_index_removed))
        test_token_id_list = list(map(_encode, token_strings))

        max_token_length = max(map(len, token_id_list + test_token_id_list))

        pad_token_id = vocab.pad()
        token_ids = torch.tensor(
            [
                token_ids_per_tweet
                + [pad_token_id] * (max_token_length - len(token_ids_per_tweet))
                for token_ids_per_tweet in token_id_list
            ]
        )
        attention_mask = (token_ids != pad_token_id).float()

        self.train_data, self.validation_data = self._train_validation_split(
            self.config.validation_size,
            TensorDataset(token_ids, attention_mask, labels),
        )

        test_token_ids = torch.tensor(
            [
                token_ids_per_tweet
                + [pad_token_id] * (max_token_length - len(token_ids_per_tweet))
                for token_ids_per_tweet in test_token_id_list
            ]
        )
        test_attention_mask = (test_token_ids != pad_token_id).float()
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bertweet(token_ids, attention_mask)
        sequence_output = outputs[0]
        return self.classifier(sequence_output)
