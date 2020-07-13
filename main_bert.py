import json
import os

import pytorch_lightning as pl
from bert_sentiment_classifier import BertSentimentClassifier
from bert_sentiment_classifier_add_data import BertSentimentClassifierAddData
from utilities.general_utilities import (
    build_comet_logger,
    build_save_path,
    get_args,
    get_bunch_config_from_json,
    initialize_trainer,
)


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)
    pl.seed_everything(config.random_seed)
    save_path = build_save_path(config)
    os.makedirs(save_path)

    save_path_dict = {"model_save_path": save_path}
    with open(args.config, "r+") as file:
        data = json.load(file)
        data.update(save_path_dict)
        file.seek(0)
        json.dump(data, file, indent=4)
    config = get_bunch_config_from_json(args.config)

    logger = build_comet_logger(save_path, config)
    logger.log_hyperparams(config)
    logger.log_hyperparams({"model_checkpoint_path": save_path})

    trainer = initialize_trainer(save_path, config, logger)

    if args.test_model_path is None:
        if config.use_augmented_data:
            model = BertSentimentClassifierAddData(config)
        else:
            model = BertSentimentClassifier(config)
        trainer.fit(model)
    else:
        if config.use_augmented_data:
            model = BertSentimentClassifierAddData.load_from_checkpoint(
                args.test_model_path, config=config
            )
        else:
            model = BertSentimentClassifier.load_from_checkpoint(
                args.test_model_path, config=config
            )
    trainer.test(model=model)


if __name__ == "__main__":
    main()
