import argparse
import html
import os
import re
import string

import pandas as pd


def preprocess(tweet: str) -> str:
    tweet = re.sub(url, "<url>", tweet)
    tweet = re.sub(user_mention, "<user>", tweet)
    tweet = tweet.lower()
    tweet = re.sub(
        "(.)\\1(\\1)+", "\\1\\1\\1", tweet
    )  # limit length of repeated letters to 3
    tweet = html.unescape(tweet)  # i.e. &quot; -> "
    tweet = re.sub(r"([,()!?])([,()!?])([,()!?])", "\\1 \\2 \\3", tweet)  # !!! -> ! ! !
    tweet = re.sub(r"([,()!?])([,()!?])", "\\1 \\2", tweet)  # !! -> ! !
    tweet = re.sub(r"([.,()!?%])(\w)", "\\1 \\2", tweet)  # .word -> . word
    tweet = re.sub(r"(\w)([.,()!?%])", "\\1 \\2", tweet)  # word. -> word .
    tweet = re.sub("([a-zA-Z])/([a-zA-Z])", "\\1 / \\2", tweet)
    tweet = " ".join(tweet.split())  # remove double whitespaces
    return tweet


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input_tweets", help="Add input tweets path")
    argparser.add_argument(
        "-o", "--output_folder", help="Save path for preprocessed tweets"
    )
    args = argparser.parse_args()

    url = re.compile(r"(?:(http[s]?://\S+)|((//)?(\w+\.)?\w+\.\w+/\S+))")
    user_mention = re.compile(r"(?:(?<!\w)@\w+\b)")

    raw_data = pd.read_csv(  # type: ignore
        args.input_tweets,
        encoding="latin-1",
        names=["label", "id", "date", "attr", "user", "tweet"],
    )
    labels = raw_data["label"]
    tweets = raw_data["tweet"]

    negative_tweets = tweets[labels == 0]
    positive_tweets = tweets[labels == 4]

    preprocessed_negative_tweets = [
        preprocess(tweet)
        for tweet in negative_tweets
        if all(c in string.printable for c in tweet)
    ]
    preprocessed_positive_tweets = [
        preprocess(tweet)
        for tweet in positive_tweets
        if all(c in string.printable for c in tweet)
    ]

    with open(os.path.join(args.output_folder, "additional_train_neg.txt"), "w") as out:
        for neg_tweet in preprocessed_negative_tweets:
            out.write(neg_tweet + "\n")

    with open(os.path.join(args.output_folder, "additional_train_pos.txt"), "w") as out:
        for positive_tweet in preprocessed_positive_tweets:
            out.write(positive_tweet + "\n")
