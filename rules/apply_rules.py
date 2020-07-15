from collections import deque
from typing import List, Tuple

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


def predict_parenthesis(tweets: List[str]) -> List[int]:
    return [classify_parenthesis(tweet) for tweet in tweets]


def classify(tweet: str, positive_pattern: str, negative_pattern: str) -> int:
    if (positive_pattern in tweet) and (negative_pattern in tweet):
        return 0
    elif positive_pattern in tweet:
        return 1
    elif negative_pattern in tweet:
        return -1
    else:
        return 0


def predict_hearts_nospace(tweets: List[str]) -> List[int]:
    return [classify(tweet, "<3", "</3") for tweet in tweets]


def predict_hearts_space(tweets: List[str]) -> List[int]:
    return [classify(tweet, "< 3", "< / 3") for tweet in tweets]


def predict_equal_sign_smileys(tweets: List[str]) -> List[int]:
    return [classify(tweet, "=)", "=(") for tweet in tweets]


def get_subsets_rule_based(
    rule_predictions: List[int], bert_predictions: List[int], labels: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    rule_predictions_rule_matched = []
    bert_predictions_rule_matched = []
    labels_rule_matched = []
    for x, y, z in zip(rule_predictions, bert_predictions, labels):
        if x != 0:
            rule_predictions_rule_matched.append(x)
            bert_predictions_rule_matched.append(y)
            labels_rule_matched.append(z)
    return (
        rule_predictions_rule_matched,
        bert_predictions_rule_matched,
        labels_rule_matched,
    )


def print_all_scores(rule_predictions: List[int], labels: List[int]) -> None:
    print("-- all tweets --")
    print("nr of rule_predictions: ", len([x for x in rule_predictions if x != 0]))
    print("nr of unknown: ", len([x for x in rule_predictions if x == 0]))
    confusion_matrix_rules = confusion_matrix(labels, rule_predictions)
    print("confusion matrix:\n", confusion_matrix_rules)


def print_rule_scores(
    rule_predictions: List[int], bert_predictions: List[int], labels: List[int],
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


def predict(tweets: List[str]) -> List[int]:
    predictions_parenthesis = predict_parenthesis(tweets)
    predictions_hearts_nospace = predict_hearts_nospace(tweets)
    predictions_hearts_space = predict_hearts_space(tweets)
    predictions_equal_sign_smileys = predict_equal_sign_smileys(tweets)
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

    return rule_predictions


def main() -> None:
    tweets = load_tweets("data/rules/validation_data.txt")
    tweets_index_removed = remove_indices_from_test_tweets(tweets)

    with open("data/rules/validation_labels.txt") as f:
        labels = [int(label) for label in f]

    # this are gonna be the real predictions of bert once finished
    bert_predictions = labels

    rule_predictions = predict(tweets_index_removed)

    (
        rule_predictions_rule_matched,
        bert_predictions_rule_matched,
        labels_rule_matched,
    ) = get_subsets_rule_based(rule_predictions, bert_predictions, labels)

    print_all_scores(rule_predictions, labels)

    print_rule_scores(
        rule_predictions_rule_matched,
        bert_predictions_rule_matched,
        labels_rule_matched,
    )


if __name__ == "__main__":
    main()
