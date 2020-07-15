from collections import deque
from typing import List

import numpy as np
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


def classify_parenthesis(tweet: str) -> int:
    return classify(remove_matching_parenthesis(tweet), ")", "(")


def classify(tweet: str, positive_pattern: str, negative_pattern: str) -> int:
    if (positive_pattern in tweet) and (negative_pattern in tweet):
        return 0
    elif positive_pattern in tweet:
        return 1
    elif negative_pattern in tweet:
        return -1
    else:
        return 0


def print_all_scores(rule_predictions: np.ndarray, labels: np.ndarray) -> None:
    print("-- all tweets --")
    print("nr of rule_predictions: ", len([x for x in rule_predictions if x != 0]))
    print("nr of unknown: ", len([x for x in rule_predictions if x == 0]))
    confusion_matrix_rules = confusion_matrix(labels, rule_predictions)
    print("confusion matrix:\n", confusion_matrix_rules)


def print_rule_scores(
    rule_predictions: np.ndarray, bert_predictions: np.ndarray, labels: np.ndarray,
) -> None:
    print("-- only tweets predicted with rules --")
    accuracy_rules = accuracy_score(labels, rule_predictions)
    print("accuracy rules: ", accuracy_rules)
    accuracy_bert = accuracy_score(labels, bert_predictions)
    print("accuracy bert: ", accuracy_bert)
    confusion_matrix_rules = confusion_matrix(labels, rule_predictions)
    print("confusion matrix rules:\n", confusion_matrix_rules)
    confusion_matrix_bert = confusion_matrix(labels, bert_predictions)
    print("confusion matrix bert:\n", confusion_matrix_bert)


def predict(tweets: List[str]) -> np.ndarray:
    predictions_parenthesis = [classify_parenthesis(tweet) for tweet in tweets]
    predictions_hearts_nospace = [classify(tweet, "<3", "</3") for tweet in tweets]
    predictions_hearts_space = [classify(tweet, "< 3", "< / 3") for tweet in tweets]
    predictions_equal_sign_smileys = [classify(tweet, "=)", "=(") for tweet in tweets]
    rule_predictions = []

    for a, b, c, d in zip(
        predictions_parenthesis,
        predictions_hearts_space,
        predictions_hearts_nospace,
        predictions_equal_sign_smileys,
    ):
        if sum([a, b, c, d]) > 0:
            rule_predictions.append(1)
        elif sum([a, b, c, d]) < 0:
            rule_predictions.append(-1)
        else:
            rule_predictions.append(0)

    return np.array(rule_predictions)


def main() -> None:
    tweets = load_tweets("data/rules/validation_data.txt")
    tweets_index_removed = remove_indices_from_test_tweets(tweets)

    labels = np.loadtxt("data/rules/validation_labels.txt", dtype=np.int)
    # this are gonna be the real predictions of bert once finished
    bert_predictions = labels

    rule_predictions = predict(tweets_index_removed)
    rule_predictions = np.array(rule_predictions)

    rule_predictions_rule_matched = rule_predictions[rule_predictions != 0]
    bert_predictions_rule_matched = bert_predictions[rule_predictions != 0]
    labels_rule_matched = labels[rule_predictions != 0]

    print_all_scores(rule_predictions, labels)

    print_rule_scores(
        rule_predictions_rule_matched,
        bert_predictions_rule_matched,
        labels_rule_matched,
    )


if __name__ == "__main__":
    main()
