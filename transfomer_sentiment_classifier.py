import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from comet_ml import Experiment
import logging
import torch
import transformers
import pytorch_lightning as pl
import nlp
from utilities.general_utilities import get_args, process_config
from bunch import Bunch


class TransformerSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super().__init__()
        self.config = config
        self.model = transformers.BertForSequenceClassification.from_pretrained(
            config.transformer_model
        )
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(
            self.config.transformer_model
        )

        def _tokenize(x):
            x["input_ids"] = tokenizer.batch_encode_plus(
                x["text"],
                max_length=self.config.max_sequence_length,
                pad_to_max_length=True,
            )["input_ids"]
            return x

        def _prepare_dataset(split):
            dataset = nlp.load_dataset(
                "imdb",
                split=f'{split}[:{self.config.batch_size if self.config.debug else f"{self.config.percent}%"}]',
            )
            dataset = dataset.map(_tokenize, batched=True)
            dataset.set_format(type="torch", columns=["input_ids", "label"])
            return dataset

        self.train_ds, self.test_ds = map(_prepare_dataset, ("train", "test"))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        (logits,) = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"])
        loss = self.loss(logits, batch["label"]).mean()
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"])
        loss = self.loss(logits, batch["label"])
        acc = (logits.argmax(-1) == batch["label"]).float()
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o["loss"] for o in outputs], 0).mean()
        acc = torch.cat([o["acc"] for o in outputs], 0).mean()
        out = {"val_loss": loss, "val_acc": acc}
        return {**out, "log": out}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=True,
        )

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )


def setup() -> [Bunch, Experiment]:
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        logging.exception(
            "You need to pass a config as argument, i.e. pass -c /path/to/config_file.json"
        )
        exit(0)

    assert config.comet_api_key is not None, "Comet api key not defined in config"
    assert (
        config.comet_project_name is not None
    ), "Comet project name not defined in config"
    assert config.comet_workspace is not None, "Comet workspace not defined in config"
    assert (
        config.use_comet_experiments is not None
    ), "Comet use_experiment flag not defined in config"

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.log_asset(args.config)
    return config, comet_experiment


def main():
    config, comet_experiment = setup()
    model = TransformerSentimentClassifier(config)
    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=config.epochs,
        fast_dev_run=config.debug,
        logger=pl.loggers.TensorBoardLogger("logs/", name="imdb", version=0),
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
