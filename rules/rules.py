from collections import deque
from typing import List

import numpy as np


class ParenthesisRule:
    def parenthesis_rule(self, tweet: str) -> int:
        return self._classify(self._remove_matching_parenthesis(tweet), ")", "(")

    def tweet_without_matching_parentheses(selfs, tweet: str) -> str:
        # new function needed
        return tweet

    def _remove_chars_at(self, indices: List[int], string: str) -> str:
        char_array = np.array(list(string))
        trimmed_char_array = np.delete(char_array, indices)
        trimmed_string = "".join(trimmed_char_array)
        return trimmed_string

    def _remove_matching_parenthesis(self, tweet: str) -> str:
        stack: deque = deque()
        matching_indices = []
        for index, char in enumerate(tweet):
            if char == "(":
                stack.append(index)
            if char == ")":
                if len(stack) > 0:
                    opening_index = stack.pop()
                    matching_indices.append(opening_index)
                    matching_indices.append(index)
        return self._remove_chars_at(matching_indices, tweet)

    def _classify(
        self, tweet: str, positive_pattern: str, negative_pattern: str
    ) -> int:
        if (positive_pattern in tweet) and (negative_pattern in tweet):
            return 0
        elif positive_pattern in tweet:
            return 1
        elif negative_pattern in tweet:
            return -1
        else:
            return 0


class HeartRule:
    def heart_rule(self, tweet: str) -> int:
        return self._classify(tweet, "< 3", "< / 3")

    def tweet_without_hearts(selfs, tweet: str) -> str:
        if ("< 3" in tweet) and ("< / 3" in tweet):
            return tweet
        elif "< 3" in tweet:
            return tweet.replace("< 3", " ")
        elif "< / 3" in tweet:
            return tweet.replace("< 3", " ")
        else:
            return tweet

    def _classify(
        self, tweet: str, positive_pattern: str, negative_pattern: str
    ) -> int:
        if (positive_pattern in tweet) and (negative_pattern in tweet):
            return 0
        elif positive_pattern in tweet:
            return 1
        elif negative_pattern in tweet:
            return -1
        else:
            return 0


class HappySad_HashtagRule:
    def happy_sad_hashtag_rule(self, tweet: str) -> int:
        return self._classify(tweet, "#happy", "#sad")

    def tweet_without_happysad_hashtags(selfs, tweet: str) -> str:
        # handle longer hastags
        return tweet

    def _classify(
        self, tweet: str, positive_pattern: str, negative_pattern: str
    ) -> int:
        if (positive_pattern in tweet) and (negative_pattern in tweet):
            return 0
        elif positive_pattern in tweet:
            return 1
        elif negative_pattern in tweet:
            return -1
        else:
            return 0


class Fml_HashtagRule:
    def fml_hashtag_rule(self, tweet: str) -> int:
        if "#fml " in tweet:
            return -1
        else:
            return 0

    def tweet_without_fml_hashtags(selfs, tweet: str) -> str:
        if "#fml " in tweet:
            return tweet.replace("#fml", "")
        else:
            return tweet


class BarSmileyRule:
    def bar_smiley_rule(self, tweet: str) -> int:
        if ": |" in tweet:
            return -1
        else:
            return 0

    def tweet_without_barsmiley(selfs, tweet: str) -> str:
        if ": |" in tweet:
            return tweet.replace(": |", "")
        else:
            return tweet


class Rule(
    ParenthesisRule, HeartRule, HappySad_HashtagRule, Fml_HashtagRule, BarSmileyRule
):
    def __init__(self) -> None:
        self.rules = [
            self.parenthesis_rule,
            self.heart_rule,
            self.happy_sad_hashtag_rule,
            self.fml_hashtag_rule,
            self.bar_smiley_rule,
        ]

    def predict(self, tweets: List[str]) -> np.ndarray:
        predictions_all_rules = [self._apply_rules(tweet) for tweet in tweets]
        predictions = [
            self._aggregate(per_tweet_predictions)
            for per_tweet_predictions in predictions_all_rules
        ]
        return np.array(predictions)

    def tweets_without_rules(self, tweets: List[str]) -> List[str]:
        for tweet in tweets:
            tweet = self.tweet_without_matching_parentheses(tweet)
            tweet = self.tweet_without_hearts(tweet)
            tweet = self.tweet_without_happysad_hashtags(tweet)
            tweet = self.tweet_without_fml_hashtags(tweet)
            tweet = self.tweet_without_barsmiley(tweet)
        return tweets

    def _apply_rules(self, tweet: str) -> List[int]:
        return [rule(tweet) for rule in self.rules]

    def _aggregate(self, predictions: List[int]) -> int:
        # return predictions[0]
        if (1 in predictions) and (-1 in predictions):
            return 0
        if sum(predictions) > 0:
            return 1
        elif sum(predictions) < 0:
            return -1
        else:
            return 0
