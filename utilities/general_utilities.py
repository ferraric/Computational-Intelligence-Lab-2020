import argparse
import json
import re
from typing import List

from bunch import Bunch


def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Add the Configuration file that has all the relevant parameters",
    )
    argparser.add_argument(
        "-t",
        "--test_model_path",
        help="Path to saved model to be used for test set prediction",
    )
    return argparser.parse_args()


def get_bunch_config_from_json(json_file_path: str) -> Bunch:
    """
    Get the config from a json file and save it as a Bunch object.
    :param json_file:
    :return: config as Bunch object:
    """
    with open(json_file_path, "r") as config_file:
        config_dict = json.load(config_file)
    return Bunch(config_dict)


def remove_indices_from_test_tweets(test_tweets: List[str]) -> List[str]:
    def _remove_index_from_test_tweet(tweet: str) -> str:
        test_tweet_format = re.compile("^[0-9]*,(.*)")
        match = test_tweet_format.match(tweet)
        if match:
            return match.group(1)
        else:
            raise ValueError("unexpected test data format")

    test_tweets_index_removed = [
        _remove_index_from_test_tweet(tweet) for tweet in test_tweets
    ]

    return test_tweets_index_removed
