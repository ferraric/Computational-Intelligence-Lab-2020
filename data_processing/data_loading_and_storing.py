from typing import List

import torch


def load_tweets(tweets_path: str) -> List[str]:
    with open(tweets_path, encoding="utf-8") as f:
        return f.read().splitlines()


def save_labels(labels: torch.Tensor, save_path: str) -> None:
    with open(save_path, "w") as out:
        for label in labels:
            label_to_save = 2 * label.item() - 1
            out.write(str(label_to_save) + "\n")


def save_tweets_in_test_format(tweets: List[str], save_path: str) -> None:
    with open(save_path, "w") as out:
        for i, tweet in enumerate(tweets, 1):
            out.write("{},{}\n".format(str(i), tweet))
