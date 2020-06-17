import inspect
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import transformers
from bunch import Bunch
from comet_ml import Experiment
from torch.utils.data import TensorDataset, random_split
from utilities.general_utilities import get_args, get_bunch_config_from_json

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))  # type: ignore
)
sys.path.insert(0, currentdir)


class TransformerSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super().__init__()
        self.config = config
        self.model = transformers.BertForSequenceClassification.from_pretrained(
            config.transformer_model
        )

    def prepare_data(self) -> None:
        tokenizer = transformers.BertTokenizer.from_pretrained(
            self.config.transformer_model
        )

        with open(self.config.negative_tweets_path) as f:
            text_lines_neg = f.read().splitlines()
        with open(self.config.positive_tweets_path) as f:
            text_lines_pos = f.read().splitlines()
        all_tweets = text_lines_neg + text_lines_pos
        labels = np.concatenate(
            (np.zeros(len(text_lines_neg)), np.ones(len(text_lines_pos)))
        )

        tokenized_input = tokenizer.batch_encode_plus(
            all_tweets,
            max_length=self.config.max_tokens_per_tweet,
            pad_to_max_length=True,
            return_token_type_ids=False,
        )
        input_ids = torch.tensor(tokenized_input["input_ids"], dtype=int)
        attention_mask = torch.tensor(tokenized_input["attention_mask"], dtype=int)
        labels = torch.tensor(labels, dtype=int)
        all_data = TensorDataset(input_ids, attention_mask, labels)

        n_val_samples = int(0.1 * len(all_data))
        n_train_samples = len(all_data) - n_val_samples
        self.train_data, self.val_data = random_split(
            all_data, [n_train_samples, n_val_samples]
        )

    def forward(self, input_ids: None) -> None:
        pass

    def training_step(self, batch: None, batch_idx: None) -> None:
        pass

    def validation_step(self, batch: None, batch_idx: None) -> None:
        pass

    def validation_epoch_end(self, outputs: None) -> None:
        pass

    def train_dataloader(self) -> None:
        pass

    def validation_dataloader(self) -> None:
        pass

    def test_data_loader(self) -> None:
        pass

    def configure_optimizers(self) -> None:
        pass


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.log_parameters(config)

    model = TransformerSentimentClassifier(config)
    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    main()
