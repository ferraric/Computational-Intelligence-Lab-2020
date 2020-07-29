import argparse
from typing import List

import numpy as np
import pandas as pd
from data_processing.data_loading_and_storing import (
    load_test_tweets,
    save_tweets_in_test_format,
)
from rule.rule_classifier import RuleClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d",
        "--validation_data_path",
        required=True,
        help="Path to the validation data",
    )
    argparser.add_argument(
        "-l",
        "--validation_labels_path",
        required=True,
        help="Path to the validation labels",
    )
    argparser.add_argument(
        "-b", "--bert_predictions_path", help="Path to the BERT predictions (csv)",
    )
    argparser.add_argument(
        "-s",
        "--save_path",
        help="Path where to save the tweets without the rule patterns",
    )
    return argparser.parse_args()


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
    args = get_args()

    tweets = load_test_tweets(args.validation_data_path)

    rule_classifier = RuleClassifier()

    if args.save_path is not None:
        tweets_without_rule_patterns = rule_classifier.remove_rule_patterns_from(tweets)
        save_tweets_in_test_format(tweets_without_rule_patterns, args.save_path)
        print("tweets saved")

    elif args.bert_predictions_path is not None:
        labels = np.loadtxt(args.validation_labels_path, dtype=np.int64)

        bert_predictions = np.loadtxt(
            args.bert_predictions_path,
            delimiter=",",
            dtype=np.int64,
            skiprows=1,
            usecols=(1,),
        )

        rule_predictions = rule_classifier.predict(tweets)

        print_confusion_matrix(
            labels,
            rule_predictions,
            label_names=["negative", "unknown", "positive"],
            title="rule based",
        )
        print_confusion_matrix(
            labels, bert_predictions, label_names=["negative", "positive"], title="bert"
        )

        rule_predictions_rule_matched = rule_predictions[rule_predictions != 0]
        bert_predictions_rule_matched = bert_predictions[rule_predictions != 0]
        labels_rule_matched = labels[rule_predictions != 0]

        print(
            "Percentage of rule matches:",
            len(rule_predictions_rule_matched) / len(rule_predictions),
        )
        accuracy_rules = accuracy_score(
            labels_rule_matched, rule_predictions_rule_matched
        )
        print("accuracy rules: ", accuracy_rules)
        accuracy_bert = accuracy_score(
            labels_rule_matched, bert_predictions_rule_matched
        )
        print("accuracy bert: ", accuracy_bert)

        print_confusion_matrix(
            labels_rule_matched,
            rule_predictions_rule_matched,
            label_names=["negative", "positive"],
            title="rule based on rule match",
        )
        print_confusion_matrix(
            labels_rule_matched,
            bert_predictions_rule_matched,
            label_names=["negative", "positive"],
            title="bert on rule match",
        )

    else:
        raise ValueError("-b and -s flag not specified")


if __name__ == "__main__":
    main()
