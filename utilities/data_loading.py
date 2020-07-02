import re
from typing import List


def remove_indices_from_test_tweets(tweets: List[str]) -> List[str]:
    def _remove_index_from_test_tweet(tweet: str) -> str:
        test_tweet_format = re.compile("^[0-9]*,(.*)")
        match = test_tweet_format.match(tweet)
        if match:
            return match.group(1)
        else:
            raise ValueError("unexpected test data format")

    return [_remove_index_from_test_tweet(tweet) for tweet in tweets]
