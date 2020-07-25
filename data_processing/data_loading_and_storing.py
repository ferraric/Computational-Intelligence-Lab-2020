import os
import re
from typing import List

import torch
from torch.utils.data import Subset


def load_tweets(tweets_path: str) -> List[str]:
    with open(tweets_path, encoding="utf-8") as f:
        return f.read().splitlines()


def load_test_tweets(tweets_path: str) -> List[str]:
    def _remove_indices_from_test_tweets(tweets: List[str]) -> List[str]:
        def _remove_index_from_test_tweet(tweet: str) -> str:
            test_tweet_format = re.compile("^[0-9]*,(.*)")
            match = test_tweet_format.match(tweet)
            if match:
                return match.group(1)
            else:
                raise ValueError("unexpected test data format")

        return [_remove_index_from_test_tweet(tweet) for tweet in tweets]

    tweets = load_tweets(tweets_path)
    return _remove_indices_from_test_tweets(tweets)


def save_labels(labels: torch.Tensor, save_path: str) -> None:
    with open(save_path, "w") as out:
        for label in labels:
            label_to_save = 2 * label.item() - 1
            out.write(str(label_to_save) + "\n")


def save_tweets_in_test_format(tweets: List[str], save_path: str) -> None:
    with open(save_path, "w") as out:
        for i, tweet in enumerate(tweets, 1):
            out.write("{},{}\n".format(str(i), tweet))


def save_validation_tweets_and_labels(
    all_tweets: List[str], labels: torch.Tensor, validation_data: Subset, save_path: str
) -> None:
    validation_indices = list(validation_data.indices)
    validation_tweets = [all_tweets[i] for i in validation_indices]
    validation_labels = labels[validation_indices]
    save_tweets_in_test_format(
        validation_tweets, os.path.join(save_path, "validation_data.txt"),
    )
    save_labels(
        validation_labels, os.path.join(save_path, "validation_labels.txt"),
    )
