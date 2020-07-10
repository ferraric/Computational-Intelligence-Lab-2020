from typing import List

from utilities.data_loading import load_tweets
from utilities.general_utilities import get_args, get_bunch_config_from_json


def remove_brackets(data: List[str]) -> List[str]:
    for i in data:
        i = i.replace("(", "").replace(")", "")
    return data


def apply_rules(data: List[str]) -> List[str]:

    data = remove_brackets(data)

    return data


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    tweets = load_tweets(config.test_tweet_path)

    tweets = apply_rules(tweets)

    # do some more evaluation
