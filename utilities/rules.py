from typing import List


def remove_brackets(data: List[str]) -> List[str]:
    for i in data:
        i = i.replace("(", "").replace(")", "")
    return data


def apply_rules(data: List[str]) -> List[str]:

    data = remove_brackets(data)

    return data
