import re
from collections import deque
from typing import List

import numpy as np


class Rule:
    def __init__(self) -> None:
        pass

    def apply_rule(self, tweet: str) -> int:
        pass

    def clean_tweet(self, tweet: str) -> str:
        pass


class PositiveNegativeRule(Rule):
    def __init__(self, pos: str, neg: str):
        self.positive_pattern = pos
        self.negative_pattern = neg

    def apply_rule(self, tweet: str) -> int:
        return self.classify(tweet)

    def clean_tweet(selfs, tweet: str) -> str:
        pass

    def classify(self, tweet: str) -> int:
        if (self.positive_pattern in tweet) and (self.negative_pattern in tweet):
            return 0
        elif self.positive_pattern in tweet:
            return 1
        elif self.negative_pattern in tweet:
            return -1
        else:
            return 0


class NegativeRule(Rule):
    def __init__(self, neg: str):
        self.negative_pattern = neg

    def apply_rule(self, tweet: str) -> int:
        return self.classify(tweet)

    def clean_tweet(self, tweet: str) -> str:
        if self.negative_pattern in tweet:
            return tweet.replace(self.negative_pattern, "")
        else:
            return tweet

    def classify(self, tweet: str) -> int:
        if self.negative_pattern in tweet:
            return -1
        else:
            return 0


class ParenthesisRule(PositiveNegativeRule):
    def apply_rule(self, tweet: str) -> int:
        return self.classify(self._remove_matching_parenthesis(tweet))

    def clean_tweet(selfs, tweet: str) -> str:
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


class HeartRule(PositiveNegativeRule):
    def apply_rule(self, tweet: str) -> int:
        return self.classify(tweet)

    def clean_tweet(selfs, tweet: str) -> str:
        if ("< 3" in tweet) and ("< / 3" in tweet):
            return tweet
        elif "< 3" in tweet:
            return tweet.replace("< 3", " ")
        elif "< / 3" in tweet:
            return tweet.replace("< 3", " ")
        else:
            return tweet


class HappySadHashtagRule(PositiveNegativeRule):
    def apply_rule(self, tweet: str) -> int:
        return self.classify(tweet)

    def clean_tweet(selfs, tweet: str) -> str:
        if ("#happy" in tweet) and ("#sad" in tweet):
            return tweet
        elif "#happy" in tweet:
            return re.sub(r"\#happy\w+", "", tweet)
        elif "#sad" in tweet:
            return re.sub(r"\#sad\w+", "", tweet)
        else:
            return tweet


class RuleClassifier(Rule):
    def __init__(self) -> None:
        self.rules = [
            HeartRule(" < 3 ", " < / 3"),
            HappySadHashtagRule("#happy", "#sad"),
            ParenthesisRule(" ) ", " ( "),
            NegativeRule("#fml "),
            NegativeRule(": | "),
        ]

    def predict(self, tweets: List[str]) -> np.ndarray:
        predictions_all_rules = [self._apply_rules(tweet) for tweet in tweets]
        predictions = [
            self._aggregate(per_tweet_predictions)
            for per_tweet_predictions in predictions_all_rules
        ]
        return np.array(predictions)

    def tweets_without_rules(self, tweets: List[str]) -> List[str]:
        pass

    def _apply_rules(self, tweet: str) -> List[int]:
        return [rule.apply_rule(tweet) for rule in self.rules]

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
