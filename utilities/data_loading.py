import re
from random import choices
from typing import List

import torch
from torch.utils.data import Dataset, Subset


def load_tweets(tweets_path: str) -> List[str]:
    with open(tweets_path, encoding="utf-8") as f:
        return f.read().splitlines()[:10]


def remove_indices_from_test_tweets(tweets: List[str]) -> List[str]:
    def _remove_index_from_test_tweet(tweet: str) -> str:
        test_tweet_format = re.compile("^[0-9]*,(.*)")
        match = test_tweet_format.match(tweet)
        if match:
            return match.group(1)
        else:
            raise ValueError("unexpected test data format")

    return [_remove_index_from_test_tweet(tweet) for tweet in tweets]


def generate_bootstrap_dataset(dataset: Dataset) -> Subset:
    dataset_size = dataset.__len__()
    sampled_indices = choices(range(dataset_size), k=dataset_size)
    return Subset(dataset, sampled_indices)


def save_labels(labels: torch.Tensor, save_path: str) -> None:
    with open(save_path, "w") as out:
        for label in labels:
            label_to_save = 2 * label.item() - 1
            out.write(str(label_to_save) + "\n")


def save_tweets_in_test_format(tweets: List[str], save_path: str) -> None:
    with open(save_path, "w") as out:
        for i, tweet in enumerate(tweets, 1):
            out.write("{},{}\n".format(str(i), tweet))
