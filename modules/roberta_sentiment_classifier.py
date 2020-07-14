import pytorch_lightning as pl
from bunch import Bunch
from modules.bert_sentiment_classifier import BertSentimentClassifier
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast


class RobertaSentimentClassifier(BertSentimentClassifier):
    def __init__(self, config: Bunch) -> None:
        pl.LightningModule.__init__(self)
        self.config = config
        self.model = RobertaForSequenceClassification.from_pretrained(
            config.pretrained_model
        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            self.config.pretrained_model
        )
        self.loss = CrossEntropyLoss()
