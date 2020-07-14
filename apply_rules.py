from typing import List, Tuple

from utilities.data_loading import load_tweets, remove_indices_from_test_tweets


def count_parentheses(data: List[str]) -> Tuple[List[float], int, int, int]:
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
            predicted_labels.append(1.0)
            pos_count += 1
        elif neg > pos:
            predicted_labels.append(-1.0)
            neg_count += 1
        else:
            predicted_labels.append(0.0)
            unknown_count += 1

    return predicted_labels, pos_count, neg_count, unknown_count


def main() -> None:

    tweets = load_tweets("data/rules/validation_data.txt")
    tweets_index_removed = remove_indices_from_test_tweets(tweets)

    predicted_labels, pos_count, neg_count, unknown_count = count_parentheses(
        tweets_index_removed
    )


if __name__ == "__main__":
    main()
