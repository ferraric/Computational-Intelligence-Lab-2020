import os
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from bert_sentiment_classifier import BertSentimentClassifier
from bunch import Bunch
from numpy.random._generator import default_rng
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import TensorDataset
from transformers import BertTokenizerFast
from utilities.general_utilities import get_args, get_bunch_config_from_json


class BertBaggingEnsembleClassifier(BertSentimentClassifier):
    def __init__(self, config: Bunch):
        super().__init__(config)

    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        negative_tweets = self._load_tweets(self.config.negative_tweets_path)
        positive_tweets = self._load_tweets(self.config.positive_tweets_path)

        labels = self._generate_labels(len(negative_tweets), len(positive_tweets))
        original_dataset_list = [negative_tweets + positive_tweets, labels.tolist()]
        original_dataset = np.array(original_dataset_list)

        bootstrap_dataset = default_rng(self.config.random_seed).choice(
            original_dataset, size=original_dataset.shape[0]
        )

        bootstrap_tweets = torch.from_numpy(bootstrap_dataset[:, 0])
        bootstrap_labels = torch.from_numpy(bootstrap_dataset[:, 1])

        train_token_ids, train_attention_mask = self._tokenize_tweets(
            tokenizer, bootstrap_tweets.tolist()
        )

        self.train_data, self.validation_data = self._train_validation_split(
            self.config.validation_size,
            TensorDataset(train_token_ids, train_attention_mask, bootstrap_labels),
        )

        test_tweets = self._load_tweets(self.config.test_tweets_path)
        test_token_ids, test_attention_mask = self._tokenize_tweets(
            tokenizer, test_tweets
        )
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)
    pl.seed_everything(config.random_seed)

    current_timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        config.model_save_directory, config.experiment_name, current_timestamp
    )
    os.makedirs(save_path)

    logger = CometLogger(
        save_dir=save_path,
        workspace=config.comet_workspace,
        project_name=config.comet_project_name,
        api_key=config.comet_api_key if config.use_comet_experiments else None,
        experiment_name=config.experiment_name,
    )
    logger.log_hyperparams(config)

    save_model_callback = ModelCheckpoint(
        os.path.join(save_path, "{epoch}-{val_loss:.2f}"), monitor="val_loss"
    )
    number_of_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        checkpoint_callback=save_model_callback,
        deterministic=True,
        fast_dev_run=config.debug,
        gpus=number_of_gpus,
        logger=logger,
        max_epochs=config.epochs,
    )

    if args.test_model_path is None:
        model = BertSentimentClassifier(config)
        trainer.fit(model)
    else:
        model = BertSentimentClassifier.load_from_checkpoint(
            args.test_model_path, config=config
        )
    trainer.test(model=model)