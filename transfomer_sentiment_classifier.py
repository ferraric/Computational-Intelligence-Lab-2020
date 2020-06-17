import inspect
import os
import sys

import pytorch_lightning as pl
import transformers
from bunch import Bunch
from comet_ml import Experiment
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
        pass

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
