import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from bert_sentiment_classifier import BertSentimentClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from utilities.general_utilities import get_args, get_bunch_config_from_json


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

    model = BertSentimentClassifier(config)
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
    trainer.fit(model)


if __name__ == "__main__":
    main()
