from typing import List

import numpy as np
import pandas as pd
from rules import RuleClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from utilities.data_loading import (
    load_tweets,
    remove_indices_from_test_tweets,
    save_tweets_in_test_format,
)


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


def remove_double_whitespaces(tweets: List[str], save_path: str) -> List[str]:
    return [" ".join(tweet.split()) for tweet in tweets]


def main() -> None:
    tweets = load_tweets("data/rules/validation_data.txt")
    tweets_index_removed = remove_indices_from_test_tweets(tweets)

    labels = np.loadtxt("data/rules/validation_labels.txt", dtype=np.int)
    # this are gonna be the real predictions of bert once finished
    bert_predictions = labels

    save_path = "data/rules/all_rules.txt"

    rule_classifier = RuleClassifier()
    rule_predictions = rule_classifier.predict(tweets_index_removed)
    tweets_without_rule_patterns = rule_classifier.remove_rule_patterns_from(
        tweets_index_removed
    )
    save_tweets_in_test_format(
        remove_indices_from_test_tweets(tweets_without_rule_patterns), save_path
    )

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
