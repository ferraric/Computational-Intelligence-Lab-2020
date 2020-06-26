import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from bunch import Bunch
from comet_ml import Experiment
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utilities.general_utilities import get_args, get_bunch_config_from_json


def generate_training_data(config: Bunch) -> Tuple[np.ndarray, np.ndarray]:
    def load_tweets_and_labels(
        neg_tweets_path: str, pos_tweets_path: str
    ) -> Tuple[List[str], np.ndarray]:
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

    def load_embeddings(embeddings_path: str) -> np.ndarray:
        with open(embeddings_path, mode="rb") as f:
            embeddings = np.load(f)["arr_0"]
        return embeddings

    vocab = load_vocabulary(config.glove_vocabulary_path)
    embeddings = load_embeddings(config.glove_embeddings_path)
    tweets, labels = load_tweets_and_labels(
        config.neg_tweets_path, config.pos_tweets_path
    )

    def construct_features(
        vocab: Dict[str, int], embeddings: np.ndarray, tweets: List[str]
    ) -> np.ndarray:
        nr_samples = len(tweets)
        nr_features = embeddings.shape[1]
        features = np.zeros((nr_samples, nr_features))

        words_in_vocab = list(vocab.keys())

        for i, tweet in enumerate(tweets):
            words_in_tweet = tweet.split()
            nr_found_embeddings = 0
            for word in words_in_tweet:
                if word in words_in_vocab:
                    nr_found_embeddings += 1
                    embedding_index = words_in_vocab.index(word)
                    features[i, :] += embeddings[embedding_index, :]
            if nr_found_embeddings > 0:
                features[i, :] /= nr_found_embeddings
        return features

    features = construct_features(vocab, embeddings, tweets)
    return features, labels


def load_test_data(test_data_path: str) -> List[str]:
    with open(test_data_path, encoding="utf-8") as f:
        test_data = f.read().splitlines()
    return test_data


def run_grid_search(
    reg_params: Dict[str, float], features: np.ndarray, labels: np.ndarray
) -> SVC:
    model = SVC()
    grid_search = GridSearchCV(estimator=model, param_grid=reg_params)
    grid_search.fit(features, labels)
    return grid_search.best_estimator_


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.set_name(config.comet_experiment_name)
    comet_experiment.log_parameters(config)

    training_features, training_labels = generate_training_data(config)
    best_model = run_grid_search(
        config.regularization_parameters, training_features, training_labels
    )

    test_data = load_test_data(config.test_data_path)
    ids = np.arange(1, len(test_data) + 1)
    predictions = best_model.predict(test_data)
    predictions_table = np.hstack((ids, predictions))

    comet_experiment.log_table(
        filename="test_predictions.csv",
        tabular_data=predictions_table,
        headers=["Id", "Prediction"],
    )


if __name__ == "__main__":
    main()
