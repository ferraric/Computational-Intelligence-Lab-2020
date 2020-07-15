from collections import deque
from typing import List

import numpy as np


class RuleClassifier:
    def __init__(self) -> None:
        self.rules = [
            self.parenthesis_rule,
            self.heart_rule_no_space,
            self.heart_rule_space,
            self.long_eyed_smiley_rule,
        ]

    def predict(self, tweets: List[str]) -> np.ndarray:
        predictions_all_rules = [self._apply_rules(tweet) for tweet in tweets]
        predictions = [
            self._aggregate(per_tweet_predictions)
            for per_tweet_predictions in predictions_all_rules
        ]

        return np.array(predictions)

    def _apply_rules(self, tweet: str) -> List[int]:
        return [rule(tweet) for rule in self.rules]

    def _aggregate(self, predictions: List[int]) -> int:
        return predictions[0]

    def parenthesis_rule(self, tweet: str) -> int:
        return self._classify(self._remove_matching_parenthesis(tweet), ")", "(")

    def heart_rule_no_space(self, tweet: str) -> int:
        return self._classify(tweet, "<3", "</3")

    def heart_rule_space(self, tweet: str) -> int:
        return self._classify(tweet, "< 3", "< / 3")

    def long_eyed_smiley_rule(self, tweet: str) -> int:
        return self._classify(tweet, "=)", "=(")

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
