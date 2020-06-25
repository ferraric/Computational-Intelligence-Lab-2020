import pickle
from typing import Any, List, Tuple

import numpy as np
from bunch import Bunch
from comet_ml import Experiment
from utilities.general_utilities import get_args, get_bunch_config_from_json


def generate_data(config: Bunch) -> None:
    def load_tweets_and_labels(
        neg_tweets_path: str, pos_tweets_path: str
    ) -> Tuple[List[str], np.Array]:
        with open(neg_tweets_path, encoding="utf-8") as f:
            neg_tweets = f.read().splitlines()
        with open(pos_tweets_path, encoding="utf-8") as f:
            pos_tweets = f.read().splitlines()
        tweets = neg_tweets + pos_tweets
        labels = np.vstack((np.zeros(len(neg_tweets)), np.ones(len(pos_tweets))))
        return tweets, labels

    def load_vocabulary(vocabulary_path: str) -> Any:
        with open(vocabulary_path, mode="rb") as f:
            vocabulary = pickle.load(f)
        return vocabulary

    def load_embeddings(embeddings_path: str) -> np.Array:
        with open(embeddings_path) as f:
            embeddings = np.load(f)["arr_0"]
        return embeddings


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


if __name__ == "__main__":
    main()
