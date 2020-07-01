import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Dict, List, Tuple, Union

import numpy as np
from bunch import Bunch
from comet_ml import Experiment
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from utilities.general_utilities import get_args, get_bunch_config_from_json


def load_tweets(tweets_path: str) -> List[str]:
    with open(tweets_path, encoding="utf-8") as f:
        return f.read().splitlines()


def load_vocabulary(vocabulary_path: str) -> List[str]:
    with open(vocabulary_path, mode="rb") as f:
        word_occurrence_dict = pickle.load(f)
        return list(word_occurrence_dict.keys())


def load_embeddings(embeddings_path: str) -> np.ndarray:
    with open(embeddings_path, mode="rb") as f:
        return np.load(f)["arr_0"]


def construct_features(config: Bunch, tweets: List[str]) -> np.ndarray:
    vocabulary = load_vocabulary(config.glove_vocabulary_path)
    embeddings = load_embeddings(config.glove_embeddings_path)

    nr_samples = len(tweets)
    nr_features = embeddings.shape[1]
    features = np.zeros((nr_samples, nr_features))

    for i, tweet in enumerate(tweets):
        words_in_tweet = tweet.split()
        nr_found_embeddings = 0
        for word in words_in_tweet:
            if word in vocabulary:
                nr_found_embeddings += 1
                embedding_index = vocabulary.index(word)
                features[i, :] += embeddings[embedding_index, :]
        if nr_found_embeddings > 0:
            features[i, :] /= nr_found_embeddings
    return features


def construct_features_parallel(config: Bunch, tweets: List[str]) -> np.ndarray:
    split_tweets = [
        a.tolist() for a in np.array_split(np.array(tweets), config.num_workers)
    ]

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        feature_chunks = executor.map(construct_features, repeat(config), split_tweets)
        return np.concatenate(tuple(feature_chunks), axis=0)


def generate_training_data(config: Bunch) -> Tuple[np.ndarray, np.ndarray]:
    def load_tweets_and_labels(
        neg_tweets_path: str, pos_tweets_path: str
    ) -> Tuple[List[str], np.ndarray]:
        neg_tweets = load_tweets(neg_tweets_path)
        pos_tweets = load_tweets(pos_tweets_path)
        all_tweets = neg_tweets + pos_tweets
        labels = np.concatenate(
            (-1 * np.ones(len(neg_tweets)), np.ones(len(pos_tweets))), axis=0
        )
        return all_tweets, labels

    tweets, labels = load_tweets_and_labels(
        config.neg_tweets_path, config.pos_tweets_path
    )

    features = construct_features_parallel(config, tweets)
    return features, labels


def generate_test_data_features(config: Bunch) -> np.ndarray:
    test_tweets = load_tweets(config.test_data_path)

    return construct_features_parallel(config, test_tweets)


def run_grid_search(
    random_seed: int,
    svm_params: Dict[str, List[Union[float, str]]],
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[SVC, float, Dict[str, List[Union[float, str]]]]:
    model = SVC()
    grid_search = GridSearchCV(
        estimator=model,
        cv=KFold(shuffle=True, random_state=random_seed),
        param_grid=svm_params,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(features, labels)
    return (
        grid_search.best_estimator_,
        grid_search.best_score_,
        grid_search.best_params_,
    )


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    np.random.seed(config.random_seed)

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.set_name(config.comet_experiment_name)
    comet_experiment.log_parameters(config)

    training_features, training_labels = generate_training_data(config)
    best_model, best_model_score, best_model_params = run_grid_search(
        config.random_seed, config.svm_parameters, training_features, training_labels
    )

    comet_experiment.log_metric("mean accuracy", best_model_score)
    comet_experiment.log_parameters(best_model_params)

    test_data_features = generate_test_data_features(config)
    ids = np.arange(1, test_data_features.shape[0] + 1)
    predictions = best_model.predict(test_data_features)
    print(best_model_score)
    print(predictions)
    predictions_table = np.hstack((ids, predictions))

    comet_experiment.log_table(
        filename="test_predictions.csv",
        tabular_data=predictions_table,
        headers=["Id", "Prediction"],
    )


if __name__ == "__main__":
    main()
