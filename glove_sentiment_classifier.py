from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from bunch import Bunch
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


class GloveSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super().__init__()

        self.config = config

        glove_embeddings_npz = np.load(config.embedding_file)
        glove_embeddings_tensor = torch.from_numpy(glove_embeddings_npz["arr_0"])
        self.embedding = torch.nn.Embedding.from_pretrained(glove_embeddings_tensor)
        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_size,
            num_layers=config.number_of_layers,
            dropout=config.dropout_rate,
        )

        self.fc = torch.nn.Linear(config.hidden_size, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        word_vectors: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embeds = self.embedding(word_vectors)
        lstm_outputs, new_hidden_state = self.lstm(embeds, hidden_state)

        lstm_stacked_outputs = lstm_outputs.contiguous().view(
            -1, self.config.hidden_size
        )
        linear_outputs = self.fc(lstm_stacked_outputs)
        logits = self.sigmoid(linear_outputs)

        return logits, new_hidden_state

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), self.config.learning_rate)
