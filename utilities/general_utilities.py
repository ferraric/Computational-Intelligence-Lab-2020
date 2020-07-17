import argparse
import json
import os
from datetime import datetime

import torch
from bunch import Bunch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger


def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Add the Configuration file that has all the relevant parameters",
    )
    argparser.add_argument(
        "-t",
        "--test_model_path",
        help="Path to saved model to be used for test set prediction",
    )
    return argparser.parse_args()


def get_bunch_config_from_json(json_file_path: str) -> Bunch:
    """
    Get the config from a json file and save it as a Bunch object.
    :param json_file:
    :return: config as Bunch object:
    """
    with open(json_file_path, "r") as config_file:
        config_dict = json.load(config_file)
    return Bunch(config_dict)


def build_save_path(config: Bunch) -> str:
    current_timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    return os.path.join(
        config.model_save_directory, config.experiment_name, current_timestamp
    )


def build_comet_logger(save_dir: str, config: Bunch) -> CometLogger:
    return CometLogger(
        save_dir=save_dir,
        workspace=config.comet_workspace,
        project_name=config.comet_project_name,
        api_key=config.comet_api_key if config.use_comet_experiments else None,
        experiment_name=config.experiment_name,
    )


def initialize_trainer(save_path: str, config: Bunch, logger: CometLogger) -> Trainer:
    save_model_callback = ModelCheckpoint(
        os.path.join(save_path, "{epoch}-{val_acc:.2f}"), monitor="val_acc"
    )
    number_of_gpus = 1 if torch.cuda.is_available() else 0
    return Trainer(
        checkpoint_callback=save_model_callback,
        deterministic=True,
        fast_dev_run=config.debug,
        gpus=number_of_gpus,
        logger=logger,
        max_epochs=config.epochs,
    )
