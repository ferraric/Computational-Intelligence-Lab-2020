import os

import pytorch_lightning as pl
from bert_sentiment_classifier import BertSentimentClassifier
from bunch import Bunch
from torch.nn import CrossEntropyLoss
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
