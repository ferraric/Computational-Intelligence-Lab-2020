import os
from argparse import Namespace

import pytorch_lightning as pl
from bunch import Bunch
from data_processing.bertweet_data_processor import BertweetDataProcessor
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from modules.bert_sentiment_classifier import BertSentimentClassifier
from torch.nn import CrossEntropyLoss
from transformers import RobertaConfig, RobertaForSequenceClassification


class BERTweet(BertSentimentClassifier):
    def __init__(self, config: Bunch) -> None:
        pl.LightningModule.__init__(self)
        self.config = config

        bpe_codes_path = os.path.join(
            config.pretrained_model_base_path, "BERTweet_base_transformers/bpe.codes",
        )
        bpe = fastBPE(Namespace(bpe_codes=bpe_codes_path))
        vocab = Dictionary()
        vocab.add_from_file(
            os.path.join(
                config.pretrained_model_base_path,
                "BERTweet_base_transformers/dict.txt",
            )
        )

        self.data_processor = BertweetDataProcessor(config, bpe, vocab)

        model_config = RobertaConfig.from_pretrained(
            os.path.join(
                config.pretrained_model_base_path,
                "BERTweet_base_transformers/config.json",
            )
        )
        self.model = RobertaForSequenceClassification.from_pretrained(
            os.path.join(
                config.pretrained_model_base_path,
                "BERTweet_base_transformers/model.bin",
            ),
            config=model_config,
        )
        self.loss = CrossEntropyLoss()
