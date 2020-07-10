import re
from random import choices
from typing import List

from torch.utils.data import Dataset, Subset


def load_tweets(tweets_path: str) -> List[str]:
    with open(tweets_path, encoding="utf-8") as f:
        return f.read().splitlines()


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


def save_labels(labels: List[int]) -> None:
    with open("data/val_labels.txt", "w") as out:
        for j in labels:
            out.write(str(j) + "\n")


def save_tweets(tweets: List[str], indices: List[int]) -> None:
    tweet_strings = [tweets[i] for i in indices]
    j = 0
    with open("data/val_data.txt", "w") as out:
        for i in tweet_strings:
            out.write(str(j) + "," + i + "\n")
