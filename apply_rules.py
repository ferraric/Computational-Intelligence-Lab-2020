from typing import List, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix
from utilities.data_loading import load_tweets, remove_indices_from_test_tweets


def matched(tweet: str) -> bool:
    diffCounter = 0
    for i in tweet:
        if i == "(":
            diffCounter += 1
        elif i == ")":
            diffCounter -= 1
    if diffCounter == 0:
        return True
    else:
        return False


# (() denn positiv
# )(( denn unknown


def count_parenthesis(data: List[str]) -> List[int]:
    rule_predictions = []
    for tweet in data:

        pos = 0
        neg = 0
        pos += tweet.count(" ) ")
        neg += tweet.count(" ( ")

        if pos > neg:
            if neg != 0:
                if matched(tweet):
                    rule_predictions.append(1)
                else:
                    rule_predictions.append(0)
            else:
                rule_predictions.append(1)
        elif neg > pos:
            if pos != 0:
                if matched(tweet):
                    rule_predictions.append(-1)
                else:
                    rule_predictions.append(0)
            else:
                rule_predictions.append(-1)
        else:
            rule_predictions.append(0)

    return rule_predictions


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


def main() -> None:
    tweets = load_tweets("data/rules/validation_data.txt")
    tweets_index_removed = remove_indices_from_test_tweets(tweets)

    with open("data/rules/validation_labels.txt") as f:
        labels = [int(label) for label in f]

    # this are gonna be the real predictions of bert once finished
    bert_predictions = labels

    rule_predictions = count_parenthesis(tweets_index_removed)

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
