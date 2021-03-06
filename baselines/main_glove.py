import os

from comet_ml import Experiment

import numpy as np
import pandas as pd
from baselines.glove_embeddings_classifier import GloveEmbeddingsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from utilities.general_utilities import (
    build_save_path,
    get_args,
    get_bunch_config_from_json,
)


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.set_name(config.experiment_name)
    comet_experiment.log_parameters(config)

    if config.model == "randomforest":
        classifier = GloveEmbeddingsClassifier(
            RandomForestClassifier(random_state=config.random_seed)
        )
    elif config.model == "logregression":
        classifier = GloveEmbeddingsClassifier(
            LogisticRegression(solver="saga", random_state=config.random_seed)
        )
    elif config.model == "decisiontree":
        classifier = GloveEmbeddingsClassifier(
            DecisionTreeClassifier(random_state=config.random_seed)
        )
    else:
        raise ValueError("chosen model not available")

    training_features, training_labels = classifier.generate_training_data(config)
    best_model, best_model_score, best_model_params = classifier.run_grid_search(
        config.random_seed, config.model_parameters, training_features, training_labels
    )

    comet_experiment.log_metric("mean accuracy", best_model_score)
    comet_experiment.log_parameters(best_model_params)

    test_data_features = classifier.generate_test_data_features(config)
    ids = np.arange(1, test_data_features.shape[0] + 1)
    predictions = best_model.predict(test_data_features)
    predictions_table = np.stack([ids, predictions], axis=-1).astype(int)

    if comet_experiment.disabled:
        save_path = build_save_path(config)
        os.makedirs(save_path)

        formatted_predictions_table = pd.DataFrame(
            predictions_table, columns=["Id", "Prediction"], dtype=np.int32,
        )
        formatted_predictions_table.to_csv(
            os.path.join(save_path, "test_predictions.csv"), index=False
        )
    else:
        comet_experiment.log_table(
            filename="test_predictions.csv",
            tabular_data=predictions_table,
            headers=["Id", "Prediction"],
        )


if __name__ == "__main__":
    main()
