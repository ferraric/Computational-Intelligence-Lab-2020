import os

import pytorch_lightning as pl
from modules.bert_pooled_classifier import BertPooledClassifier
from utilities.general_utilities import (
    build_comet_logger,
    build_save_path,
    find_model_checkpoint_path_in,
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

    logger = build_comet_logger(save_path, config)
    logger.log_hyperparams(config)
    logger.log_hyperparams({"model_checkpoint_path": save_path})

    trainer = initialize_trainer(save_path, config, logger)

    if args.test_model_path is None:
        model = BertPooledClassifier(config)
        trainer.fit(model)

        best_model_checkpoint_path = find_model_checkpoint_path_in(save_path)
    else:
        best_model_checkpoint_path = args.test_model_path

    model = BertPooledClassifier.load_from_checkpoint(
        best_model_checkpoint_path, config=config
    )
    trainer.test(model=model)


if __name__ == "__main__":
    main()
