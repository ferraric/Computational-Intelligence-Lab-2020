import re
from collections import deque
from typing import List

import numpy as np


class Rule:
    def __init__(self) -> None:
        pass

    def apply(self, tweet: str) -> int:
        pass

    def remove_rule_pattern_from(self, tweet: str) -> str:
        pass


class PositiveNegativeRule(Rule):
    def __init__(self, positive_pattern: str, negative_pattern: str):
        self.positive_pattern = positive_pattern
        self.negative_pattern = negative_pattern

    def apply(self, tweet: str) -> int:
        if (self.positive_pattern in tweet) and (self.negative_pattern in tweet):
            return 0
        elif self.positive_pattern in tweet:
            return 1
        elif self.negative_pattern in tweet:
            return -1
        else:
            return 0

    def remove_rule_pattern_from(self, tweet: str) -> str:
        if (self.positive_pattern in tweet) and (self.negative_pattern in tweet):
            return tweet
        elif self.positive_pattern in tweet:
            return tweet.replace(self.positive_pattern, "")
        elif self.negative_pattern in tweet:
            return tweet.replace(self.negative_pattern, "")
        else:
            return tweet


class NegativeRule(Rule):
    def __init__(self, negative_pattern: str):
        self.negative_pattern = negative_pattern

    def apply(self, tweet: str) -> int:
        if self.negative_pattern in tweet:
            return -1
        else:
            return 0

    def remove_rule_pattern_from(self, tweet: str) -> str:
        if self.negative_pattern in tweet:
            return tweet.replace(self.negative_pattern, "")
        else:
            return tweet


class ParenthesisRule(PositiveNegativeRule):
    def apply(self, tweet: str) -> int:
        return super().apply(self._remove_matching_parenthesis(tweet))

    def remove_rule_pattern_from(selfs, tweet: str) -> str:
        # TODO

        return tweet

    def _remove_chars_at(self, indices: List[int], string: str) -> str:
        char_array = np.array(list(string))
        trimmed_char_array = np.delete(char_array, indices)
        trimmed_string = "".join(trimmed_char_array)
        return trimmed_string

    def _get_matching_parenthesis(self, tweet: str) -> List[int]:
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
        return matching_indices

    def _remove_matching_parenthesis(self, tweet: str) -> str:
        matching_indices = self._get_matching_parenthesis(tweet)
        return self._remove_chars_at(matching_indices, tweet)

    def _get_char_indices(self, tweet: str, char: str) -> List[int]:
        indices = []
        for n in range(len(tweet)):
            if tweet[n] == char:
                indices.append(n)
        return indices

    # alli indices vo allne chlammere
    # remove all matching
    # indices vo set diff


class HappySadHashtagRule(PositiveNegativeRule):
    def remove_rule_pattern_from(self, tweet: str) -> str:
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
            PositiveNegativeRule(" < 3 ", " < / 3"),
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

    def remove_rule_patterns_from(self, tweets: List[str]) -> List[str]:
        pass

    def _apply_rules(self, tweet: str) -> List[int]:
        return [rule.apply(tweet) for rule in self.rules]

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
