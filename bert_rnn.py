import pytorch_lightning as pl
import torch
from bert_sentiment_classifier import BertSentimentClassifier
from bunch import Bunch
from torch.nn import LSTM, CrossEntropyLoss, Dropout, Linear
from transformers import BertConfig, BertModel


class BertRNNClassifier(BertSentimentClassifier):
    def __init__(self, config: Bunch) -> None:
        pl.LightningModule.__init__(self)
        self.config = config
        model_config = BertConfig.from_pretrained(config.pretrained_model)
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.rnn = LSTM(
            input_size=model_config.hidden_size,
            hidden_size=model_config.hidden_size,
            batch_first=True,
        )
        self.dropout = Dropout(model_config.hidden_dropout_prob)
        self.classifier = Linear(model_config.hidden_size, model_config.num_labels)
        self.loss = CrossEntropyLoss()

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(token_ids, attention_mask)
        token_embeddings = outputs[0]
        rnn_out, (rnn_hidden_state, rnn_last_cell_state) = self.rnn(token_embeddings)
        rnn_hidden_state = rnn_hidden_state.squeeze()
        dropout_rnn_hidden_state = self.dropout(rnn_hidden_state)
        return self.classifier(dropout_rnn_hidden_state)
