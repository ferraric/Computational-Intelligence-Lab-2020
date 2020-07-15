from collections import deque
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from utilities.data_loading import load_tweets, remove_indices_from_test_tweets


def _remove_chars_at(indices: List[int], string: str) -> str:
    char_array = np.array(list(string))
    trimmed_char_array = np.delete(char_array, indices)
    trimmed_string = "".join(trimmed_char_array)
    return trimmed_string


def remove_matching_parenthesis(tweet: str) -> str:
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
    return _remove_chars_at(matching_indices, tweet)


def parenthesis_rule(tweet: str) -> int:
    return classify(remove_matching_parenthesis(tweet), ")", "(")


def heart_rule_no_space(tweet: str) -> int:
    return classify(tweet, "<3", "</3")


def heart_rule_space(tweet: str) -> int:
    return classify(tweet, "< 3", "< / 3")


def long_eyed_smiley_rule(tweet: str) -> int:
    return classify(tweet, "=)", "=(")


def classify(tweet: str, positive_pattern: str, negative_pattern: str) -> int:
    if (positive_pattern in tweet) and (negative_pattern in tweet):
        return 0
    elif positive_pattern in tweet:
        return 1
    elif negative_pattern in tweet:
        return -1
    else:
        return 0


def apply_rules(tweet: str, rules: Sequence[Callable[[str], int]]) -> List[int]:
    return [rule(tweet) for rule in rules]


def aggregate(predictions: List[int]) -> int:
    return predictions[0]


def predict(tweets: List[str], rules: Sequence[Callable[[str], int]]) -> np.ndarray:
    predictions_all_rules = [apply_rules(tweet, rules) for tweet in tweets]
    predictions = [
        aggregate(per_tweet_predictions)
        for per_tweet_predictions in predictions_all_rules
    ]

    return np.array(predictions)


def print_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str], title: str
) -> None:
    print(title)
    print(
        pd.DataFrame(
            confusion_matrix(y_true, y_pred), index=label_names, columns=label_names
        )
    )
    print()


def main() -> None:
    tweets = load_tweets("data/rules/validation_data.txt")
    tweets_index_removed = remove_indices_from_test_tweets(tweets)

    labels = np.loadtxt("data/rules/validation_labels.txt", dtype=np.int)
    # this are gonna be the real predictions of bert once finished
    bert_predictions = labels

    rules = [
        parenthesis_rule,
        heart_rule_no_space,
        heart_rule_space,
        long_eyed_smiley_rule,
    ]
    rule_predictions = predict(tweets_index_removed, rules)

    print_confusion_matrix(
        labels,
        rule_predictions,
        label_names=["positive", "unknown", "negative"],
        title="rule based",
    )
    print_confusion_matrix(
        labels, bert_predictions, label_names=["positive", "negative"], title="bert"
    )

    rule_predictions_rule_matched = rule_predictions[rule_predictions != 0]
    bert_predictions_rule_matched = bert_predictions[rule_predictions != 0]
    labels_rule_matched = labels[rule_predictions != 0]

    print(
        "Percentage of rule matches:",
        len(rule_predictions_rule_matched) / len(rule_predictions),
    )
    accuracy_rules = accuracy_score(labels_rule_matched, rule_predictions_rule_matched)
    print("accuracy rules: ", accuracy_rules)
    accuracy_bert = accuracy_score(labels_rule_matched, bert_predictions_rule_matched)
    print("accuracy bert: ", accuracy_bert)

    print_confusion_matrix(
        labels_rule_matched,
        rule_predictions_rule_matched,
        label_names=["positive", "negative"],
        title="rule based on rule match",
    )
    print_confusion_matrix(
        labels_rule_matched,
        bert_predictions_rule_matched,
        label_names=["positive", "negative"],
        title="bert on rule match",
    )


if __name__ == "__main__":
    main()
