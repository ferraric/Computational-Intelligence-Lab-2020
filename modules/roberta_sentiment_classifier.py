import pytorch_lightning as pl
from bunch import Bunch
from data_processing.data_processor import DataProcessor
from data_processing.pretrained_tokenizer import PreTrainedTokenizer
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
        roberta_tokenizer = RobertaTokenizerFast.from_pretrained(
            self.config.pretrained_model
        )
        tokenizer = PreTrainedTokenizer(
            roberta_tokenizer, self.config.max_tokens_per_tweet
        )
        self.data_processor = DataProcessor(config, tokenizer)
        self.loss = CrossEntropyLoss()
