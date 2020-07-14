import pytorch_lightning as pl
import torch
from bunch import Bunch
from modules.bert_sentiment_classifier import BertSentimentClassifier
from torch.nn import CrossEntropyLoss, Dropout, Linear
from transformers import BertConfig, BertModel


class BertPooledClassifier(BertSentimentClassifier):
    def __init__(self, config: Bunch) -> None:
        pl.LightningModule.__init__(self)
        self.config = config
        model_config = BertConfig.from_pretrained(config.pretrained_model)
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.dropout = Dropout(model_config.hidden_dropout_prob)
        self.classifier = Linear(model_config.hidden_size, model_config.num_labels)
        self.loss = CrossEntropyLoss()

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(token_ids, attention_mask)
        token_embeddings = outputs[0]
        avg_token_embedding = torch.mean(token_embeddings, dim=1)
        avg_token_embedding = self.dropout(avg_token_embedding)
        return self.classifier(avg_token_embedding)
