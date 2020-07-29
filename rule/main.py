import argparse
from typing import List

import numpy as np
import pandas as pd
from data_processing.data_loading_and_storing import load_test_tweets
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
        "-p", "--predictions_path", help="Path to the predictions (csv)",
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

    labels = np.loadtxt(args.validation_labels_path, dtype=np.int64)

    model_predictions = np.loadtxt(
        args.predictions_path, delimiter=",", dtype=np.int64, skiprows=1, usecols=(1,),
    )

    rule_predictions = rule_classifier.predict(tweets)

    print_confusion_matrix(
        labels,
        rule_predictions,
        label_names=["Negative", "Unknown", "Positive"],
        title="Rule Classifier",
    )
    print_confusion_matrix(
        labels,
        model_predictions,
        label_names=["Negative", "Positive"],
        title="Model predictions",
    )

    rule_predictions_rule_matched = rule_predictions[rule_predictions != 0]
    model_predictions_rule_matched = model_predictions[rule_predictions != 0]
    labels_rule_matched = labels[rule_predictions != 0]

    print(
        "Percentage of rule matches:",
        len(rule_predictions_rule_matched) / len(rule_predictions),
    )
    accuracy_rules = accuracy_score(labels_rule_matched, rule_predictions_rule_matched)
    print("Accuracy Rule Classifier: ", accuracy_rules)
    accuracy_model_predictions = accuracy_score(
        labels_rule_matched, model_predictions_rule_matched
    )
    print("Accuracy model predictions: ", accuracy_model_predictions)

    print_confusion_matrix(
        labels_rule_matched,
        rule_predictions_rule_matched,
        label_names=["Negative", "Positive"],
        title="Rule Classifier on Rule Subset",
    )
    print_confusion_matrix(
        labels_rule_matched,
        model_predictions_rule_matched,
        label_names=["Negative", "Positive"],
        title="Model predictions on Rule Subset",
    )


if __name__ == "__main__":
    main()
