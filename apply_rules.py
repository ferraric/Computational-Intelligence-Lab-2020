from typing import List, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix
from utilities.data_loading import load_tweets, remove_indices_from_test_tweets


def count_parentheses(data: List[str]) -> Tuple[List[int], int, int, int]:
    predicted_labels = []
    pos_count = 0
    neg_count = 0
    unknown_count = 0
    for tweet in data:
        pos = 0
        neg = 0
        pos += tweet.count(" ) ")
        neg += tweet.count(" ( ")

        if pos > neg:
            predicted_labels.append(1)
            pos_count += 1
        elif neg > pos:
            predicted_labels.append(-1)
            neg_count += 1
        else:
            predicted_labels.append(0)
            unknown_count += 1

    return predicted_labels, pos_count, neg_count, unknown_count


def get_subsets_rule_based(
    rule_predictions: List[int], bert_predictions: List[int], labels: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    rule_predictions_rule_subset = []
    bert_predictions_rule_subset = []
    labels_rule_subset = []
    for x, y, z in zip(rule_predictions, bert_predictions, labels):
        if x != 0:
            rule_predictions_rule_subset.append(x)
            bert_predictions_rule_subset.append(y)
            labels_rule_subset.append(z)
    return (
        rule_predictions_rule_subset,
        bert_predictions_rule_subset,
        labels_rule_subset,
    )


def print_all_scores(
    count: int, unknown_count: int, predictions: List[int], labels: List[int],
) -> None:
    print("-- all tweets --")
    print("nr of predictions: ", count)
    print("nr of unknown: ", unknown_count)
    confusion_matrix_rules = confusion_matrix(labels, predictions)
    print("confusion matrix: ", confusion_matrix_rules)


def print_rule_scores(
    predictions: List[int], bert_labels: List[int], labels: List[int],
) -> None:
    print("-- only tweets predicted with rules --")
    accuracy_rules = accuracy_score(labels, predictions)
    print("accuracy rules: ", accuracy_rules)
    accuracy_bert = accuracy_score(labels, bert_labels)
    print("accuracy bert: ", accuracy_bert)
    confusion_matrix_rules = confusion_matrix(labels, predictions)
    print("confusion matrix: ", confusion_matrix_rules)


def main() -> None:

    tweets = load_tweets("data/rules/validation_data.txt")
    tweets_index_removed = remove_indices_from_test_tweets(tweets)

    with open("data/rules/validation_labels.txt") as f:
        labels = [int(x) for x in f]

    # this are gonna be the real predictions of bert once finished
    bert_predictions = labels

    rule_predictions, pos_count, neg_count, unknown_count = count_parentheses(
        tweets_index_removed
    )

    (
        rule_predictions_rule_subset,
        bert_predictions_rule_subset,
        labels_rule_subset,
    ) = get_subsets_rule_based(rule_predictions, bert_predictions, labels)

    print_all_scores(
        pos_count + neg_count, unknown_count, rule_predictions, labels,
    )

    print_rule_scores(
        rule_predictions_rule_subset, bert_predictions_rule_subset, labels_rule_subset
    )


if __name__ == "__main__":
    main()
