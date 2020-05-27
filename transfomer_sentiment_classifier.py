import os
import sys
import inspect
from comet_ml import Experiment
import logging
import torch
import transformers
import pytorch_lightning as pl
import nlp
from utilities.general_utilities import get_args, get_bunch_config_from_json
from bunch import Bunch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


class TransformerSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super().__init__()
        self.config = config
        self.model = transformers.BertForSequenceClassification.from_pretrained(
            config.transformer_model
        )
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def prepare_data(self):
        pass

    def forward(self, input_ids):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def train_dataloader(self):
        pass

    def validation_dataloader(self):
        pass

    def test_data_loader(self):
        pass

    def configure_optimizers(self):
        pass


def setup_comet_logger(config: Bunch) -> Experiment:
    assert config.comet_api_key is not None, "Comet api key not defined in config"
    assert (
        config.comet_project_name is not None
    ), "Comet project name not defined in config"
    assert config.comet_workspace is not None, "Comet workspace not defined in config"
    assert (
        config.use_comet_experiments is not None
    ), "Comet use_experiment flag not defined in config"
    assert (
        config.experiment_name is not None
    ), "Comet experiment name is not defined in config"

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    return comet_experiment


def main():
    try:
        args = get_args()
        config = get_bunch_config_from_json(args.config)
    except RuntimeError:
        logging.exception(
            "You need to pass a config as argument, i.e. pass -c /path/to/config_file.json"
        )
        exit(0)
    comet_experiment = setup_comet_logger(config)
    comet_experiment.log_asset(args.config)
    use_gpus = 1 if torch.cuda.is_available() else 0

    model = TransformerSentimentClassifier(config)
    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=use_gpus,
        max_epochs=config.epochs,
        fast_dev_run=config.debug,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
