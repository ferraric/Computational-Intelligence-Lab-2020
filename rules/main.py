from typing import List

import numpy as np
import pandas as pd
from rules import RuleClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from utilities.data_loading import load_tweets, remove_indices_from_test_tweets


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

    rule_predictions = RuleClassifier(tweets_index_removed).predictions

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
