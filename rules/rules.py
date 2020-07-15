from collections import deque
from typing import Callable, List, Sequence

import numpy as np


def get_all_rules() -> Sequence[Callable[[str], int]]:
    return [
        parenthesis_rule,
        heart_rule_no_space,
        heart_rule_space,
        long_eyed_smiley_rule,
    ]


def parenthesis_rule(tweet: str) -> int:
    return _classify(_remove_matching_parenthesis(tweet), ")", "(")


def heart_rule_no_space(tweet: str) -> int:
    return _classify(tweet, "<3", "</3")


def heart_rule_space(tweet: str) -> int:
    return _classify(tweet, "< 3", "< / 3")


def long_eyed_smiley_rule(tweet: str) -> int:
    return _classify(tweet, "=)", "=(")


def _remove_chars_at(indices: List[int], string: str) -> str:
    char_array = np.array(list(string))
    trimmed_char_array = np.delete(char_array, indices)
    trimmed_string = "".join(trimmed_char_array)
    return trimmed_string


def _remove_matching_parenthesis(tweet: str) -> str:
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


def _classify(tweet: str, positive_pattern: str, negative_pattern: str) -> int:
    if (positive_pattern in tweet) and (negative_pattern in tweet):
        return 0
    elif positive_pattern in tweet:
        return 1
    elif negative_pattern in tweet:
        return -1
    else:
        return 0
