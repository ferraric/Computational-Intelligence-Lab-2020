import html
import os

import pandas as pd

data_path = "../data"

raw_data_path = os.path.join(data_path, "training.1600000.processed.noemoticon.csv")
raw_data = pd.read_csv(  # type: ignore
    raw_data_path,
    encoding="latin-1",
    names=["label", "id", "date", "attr", "user", "tweet"],
)
labels = raw_data["label"]
tweets = raw_data["tweet"]

negative_tweets = tweets[labels == 0]
positive_tweets = tweets[labels == 4]


def preprocess(tweet: str) -> str:
    tweet = html.unescape(tweet)
    return tweet


preprocessed_negative_tweets = [preprocess(tweet) for tweet in negative_tweets]
preprocessed_positive_tweets = [preprocess(tweet) for tweet in positive_tweets]

with open(os.path.join(data_path, "additional_train_neg.txt"), "w") as out:
    for neg_tweet in preprocessed_negative_tweets:
        out.write(neg_tweet + "\n")

with open(os.path.join(data_path, "additional_train_pos.txt"), "w") as out:
    for positive_tweet in preprocessed_positive_tweets:
        out.write(positive_tweet + "\n")
