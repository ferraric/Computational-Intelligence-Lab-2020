import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from bunch import Bunch
from sklearn.model_selection import GridSearchCV, KFold
from utilities.data_loading import load_tweets, remove_indices_from_test_tweets


class GloveEmbeddingsClassifier:
    def __init__(self, model: Any):
        self.model = model

    def load_vocabulary(self, vocabulary_path: str) -> List[str]:
        with open(vocabulary_path, mode="rb") as f:
            word_occurrence_dict = pickle.load(f)
            return list(word_occurrence_dict.keys())

    def load_embeddings(self, embeddings_path: str) -> np.ndarray:
        with open(embeddings_path, mode="rb") as f:
            return np.load(f)["arr_0"]

    def construct_features(
        self, glove_vocabulary_path: str, glove_embeddings_path: str, tweets: List[str]
    ) -> np.ndarray:
        vocabulary = self.load_vocabulary(glove_vocabulary_path)
        embeddings = self.load_embeddings(glove_embeddings_path)

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

    def construct_features_parallel(
        self, config: Bunch, tweets: List[str]
    ) -> np.ndarray:
        split_tweets = [
            a.tolist() for a in np.array_split(np.array(tweets), config.num_workers)
        ]

        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            feature_chunks = executor.map(
                self.construct_features,
                repeat(config.glove_vocabulary_path),
                repeat(config.glove_embeddings_path),
                split_tweets,
            )
            return np.concatenate(tuple(feature_chunks), axis=0)

    def generate_training_data(self, config: Bunch) -> Tuple[np.ndarray, np.ndarray]:
        def load_tweets_and_labels(
            neg_tweets_path: str, pos_tweets_path: str
        ) -> Tuple[List[str], np.ndarray]:
            neg_tweets = load_tweets(neg_tweets_path)
            pos_tweets = load_tweets(pos_tweets_path)
            all_tweets = neg_tweets + pos_tweets
            tweet_labels = np.concatenate(
                (-1 * np.ones(len(neg_tweets)), np.ones(len(pos_tweets))), axis=0
            )
            return all_tweets, tweet_labels

        tweets, labels = load_tweets_and_labels(
            config.neg_tweets_path, config.pos_tweets_path
        )

        features = self.construct_features_parallel(config, tweets)
        return features, labels

    def generate_test_data_features(self, config: Bunch) -> np.ndarray:
        test_tweets = load_tweets(config.test_data_path)

        test_tweets_index_removed = remove_indices_from_test_tweets(test_tweets)

        return self.construct_features_parallel(config, test_tweets_index_removed)

    def run_grid_search(
        self,
        random_seed: int,
        model_params: Dict[str, List[Union[float, str]]],
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[Any, float, Dict[str, List[Union[float, str]]]]:
        grid_search = GridSearchCV(
            estimator=self.model,
            cv=KFold(shuffle=True, random_state=random_seed),
            param_grid=model_params,
            n_jobs=-1,
            verbose=100,
        )
        grid_search.fit(features, labels)
        return (
            grid_search.best_estimator_,
            grid_search.best_score_,
            grid_search.best_params_,
        )
