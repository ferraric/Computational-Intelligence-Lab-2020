from collections import deque
from typing import List

import numpy as np


class Rule:
    def apply(self, tweet: str) -> int:
        raise NotImplementedError()

    def remove_rule_pattern_from(self, tweet: str) -> str:
        raise NotImplementedError()

    def remove_double_whitespaces(self, tweet: str) -> str:
        return " ".join(tweet.split())


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
        if self.apply(tweet) == 0:
            tweet_without_rule_pattern = tweet
        elif self.apply(tweet) == 1:
            tweet_without_rule_pattern = tweet.replace(self.positive_pattern, "")
        elif self.apply(tweet) == -1:
            tweet_without_rule_pattern = tweet.replace(self.negative_pattern, "")
        else:
            raise ValueError("rule application returned unexpected value")

        return self.remove_double_whitespaces(tweet_without_rule_pattern)


class NegativeRule(Rule):
    def __init__(self, negative_pattern: str):
        self.negative_pattern = negative_pattern

    def apply(self, tweet: str) -> int:
        if self.negative_pattern in tweet:
            return -1
        else:
            return 0

    def remove_rule_pattern_from(self, tweet: str) -> str:
        if self.apply(tweet) == 0:
            tweet_without_rule_pattern = tweet
        elif self.apply(tweet) == -1:
            tweet_without_rule_pattern = tweet.replace(self.negative_pattern, "")
        else:
            raise ValueError("rule application returned unexpected value")

        return self.remove_double_whitespaces(tweet_without_rule_pattern)


class ParenthesisRule(PositiveNegativeRule):
    def apply(self, tweet: str) -> int:
        return super().apply(self.remove_matching_parenthesis(tweet))

    def remove_rule_pattern_from(self, tweet: str) -> str:
        if self.apply(tweet) != 0:
            parenthesis_indices = self._get_parenthesis_indices(tweet)
            matching_parenthesis_indices = self._get_indices_of_matching_parenthesis(
                tweet
            )
            unmatching_parentheses_indices = [
                index
                for index in parenthesis_indices
                if index not in matching_parenthesis_indices
            ]
            tweet_without_rule_pattern = self._remove_chars_at(
                unmatching_parentheses_indices, tweet
            )
        else:
            tweet_without_rule_pattern = tweet

        return self.remove_double_whitespaces(tweet_without_rule_pattern)

    def _remove_chars_at(self, indices: List[int], string: str) -> str:
        char_array = np.array(list(string))
        trimmed_char_array = np.delete(char_array, indices)
        trimmed_string = "".join(trimmed_char_array)
        return trimmed_string

    def _get_indices_of_matching_parenthesis(self, tweet: str) -> List[int]:
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

    def remove_matching_parenthesis(self, tweet: str) -> str:
        matching_indices = self._get_indices_of_matching_parenthesis(tweet)
        return self._remove_chars_at(matching_indices, tweet)

    def _get_parenthesis_indices(self, tweet: str) -> List[int]:
        indices = []
        for i, char in enumerate(tweet):
            if (char == self.positive_pattern) or (char == self.negative_pattern):
                indices.append(i)
        return indices


class HappySadHashtagRule(PositiveNegativeRule):
    def remove_rule_pattern_from(self, tweet: str) -> str:
        if self.apply(tweet) == 0:
            tweet_without_rule_pattern = tweet
        elif self.apply(tweet) == 1:
            tweet_without_rule_pattern = " ".join(
                word
                for word in tweet.split(" ")
                if not word.startswith(self.positive_pattern)
            )
        elif self.apply(tweet) == -1:
            tweet_without_rule_pattern = " ".join(
                word
                for word in tweet.split(" ")
                if not word.startswith(self.negative_pattern)
            )
        else:
            raise ValueError("rule application returned unexpected value")

        return self.remove_double_whitespaces(tweet_without_rule_pattern)


class RuleClassifier(Rule):
    def __init__(self) -> None:
        self.rules = [
            ParenthesisRule(")", "("),
            PositiveNegativeRule("< 3 ", "< / 3"),
            HappySadHashtagRule("#happ", "#sad"),
            NegativeRule("#fml"),
            NegativeRule(": |"),
        ]

    def predict(self, tweets: List[str]) -> np.ndarray:
        predictions_all_rules = [self._apply_rules(tweet) for tweet in tweets]
        predictions = [
            self._aggregate(per_tweet_predictions)
            for per_tweet_predictions in predictions_all_rules
        ]
        return np.array(predictions)

    def _apply_rules(self, tweet: str) -> List[int]:
        return [rule.apply(tweet) for rule in self.rules]

    def _aggregate(self, predictions: List[int]) -> int:
        if (1 in predictions) and (-1 in predictions):
            return 0
        if sum(predictions) > 0:
            return 1
        elif sum(predictions) < 0:
            return -1
        else:
            return 0

    def remove_rule_patterns_from(self, tweets: List[str]) -> List[str]:
        tweets_without_rule_patterns = []
        for tweet in tweets:
            tweet_without_rule_pattern = tweet
            for rule in self.rules:
                tweet_without_rule_pattern = rule.remove_rule_pattern_from(
                    tweet_without_rule_pattern
                )
            tweets_without_rule_patterns.append(tweet_without_rule_pattern)

        assert len(tweets) == len(tweets_without_rule_patterns)
        return tweets_without_rule_patterns
