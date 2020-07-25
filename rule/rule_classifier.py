from typing import List

import numpy as np
from rule.rules import ParenthesisRule, Rule


class RuleClassifier(Rule):
    def __init__(self) -> None:
        self.rules = [
            ParenthesisRule(")", "("),
            # PositiveNegativeRule("< 3 ", "< / 3"),
            # HappySadHashtagRule("#happ", "#sad"),
            # NegativeRule("#fml"),
            # NegativeRule(": |"),
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
