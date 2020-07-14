from typing import List, Tuple

from utilities.data_loading import load_tweets, remove_indices_from_test_tweets


def count_parentheses(data: List[str]) -> Tuple[List[int], int, int, int]:
    predicted_labels = []
    pos_count = 0
    neg_count = 0
    unknown_count = 0
    for tweet in data:
        pos = 0
        neg = 0
        pos += tweet.count(" ( ")
        neg += tweet.count(" ) ")

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


def get_rule_based_labels(
    predicted_labels: List[int], bert_labels: List[int], true_labels: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    predicted_labels_rule_based = []
    bert_labels_rule_based = []
    true_labels_rule_based = []
    for x, y, z in zip(predicted_labels, bert_labels, true_labels):
        if x != 0:
            predicted_labels_rule_based.append(x)
            bert_labels_rule_based.append(y)
            true_labels_rule_based.append(z)
    return predicted_labels_rule_based, bert_labels_rule_based, true_labels_rule_based


def main() -> None:

    tweets = load_tweets("data/rules/validation_data.txt")
    tweets_index_removed = remove_indices_from_test_tweets(tweets)

    with open("data/rules/validation_labels.txt") as f:
        true_labels = [int(x) for x in f]

    bert_labels = (
        true_labels  # this are gonna be the real predictions of bert once finished
    )

    predicted_labels, pos_count, neg_count, unknown_count = count_parentheses(
        tweets_index_removed
    )

    (
        predicted_labels_rule_based,
        bert_labels_rule_based,
        true_labels_rule_based,
    ) = get_rule_based_labels(predicted_labels, bert_labels, true_labels)


if __name__ == "__main__":
    main()
