import os
import pandas as pd
import numpy as np


def parse_train_files(data_path: str, use_small_dataset: bool, use_large_dataset: bool, parsing_function=None):
    """
    This function takes a text file with positive sentiment sentences and a file with negative sentiment sentences,
    parses the text and puts it into a dataframe, and returns both dataframes concatenated to each other. Alternatively
    it can also take 2 text files per sentiment class.
    It returns a dataframe with each row corresponding to one text line in the input file. It has a column that can be
    retrieved by dataframe["text"] and a column with the sentiment label can be retrieved by df["label"]. The
    label for a positive sentiment is 1 and for a negative sentiment 0. When parsing the input text, one can
    parse it with the function "parsing_function" given as input. "parsing_function" takes a string as an input and
    returns a string.
    Note the goal of the function "parsing_function" is NOT to convert the words to numeric ID's. This should be done later.
    The goal is to maybe replace certain words such as <url>.
    :param data_path:
    :param use_small_dataset:
    :param use_large_dataset:
    :param parsing_function: Function taking a string as input and returning as string.
    :return: a dataframe with the corresponding positive and negative train data
    """
    assert os.path.isdir(data_path)
    if(parsing_function != None):
        assert callable(parsing_function)
    if (use_small_dataset and use_large_dataset):
        positive_train_name = "train_pos.txt"
        positive_train_name_large = "train_pos_full.txt"
        negative_train_name = "train_neg.txt"
        negative_train_name_large = "train_neg_full.txt"

        positive_train_file = os.path.join(data_path, positive_train_name)
        positive_train_file_large = os.path.join(data_path, positive_train_name_large)
        negative_train_file = os.path.join(data_path, negative_train_name)
        negative_train_file_large = os.path.join(data_path, negative_train_name_large)
        assert os.path.isfile(positive_train_file)
        assert os.path.isfile(positive_train_file_large)
        assert os.path.isfile(negative_train_file)
        assert os.path.isfile(negative_train_file_large)

        positive_train_dataframe = _create_dataframe_of_data_and_label(positive_train_file, sentiment=1, parsing_function=parsing_function)
        positive_train_large_dataframe = _create_dataframe_of_data_and_label(positive_train_file_large, sentiment=1, parsing_function=parsing_function)
        negative_train_dataframe = _create_dataframe_of_data_and_label(text_file_path=negative_train_file, sentiment=0, parsing_function=parsing_function)
        negative_train_large_dataframe = _create_dataframe_of_data_and_label(text_file_path=negative_train_file_large,
                                                                             sentiment=0, parsing_function=parsing_function)

        all_train_dataframe = positive_train_dataframe.append(positive_train_large_dataframe)
        all_train_dataframe = all_train_dataframe.append(negative_train_dataframe)
        all_train_dataframe = all_train_dataframe.append(negative_train_large_dataframe)
        return all_train_dataframe

    elif (use_small_dataset):
        positive_train_name = "train_pos.txt"
        negative_train_name = "train_neg.txt"
    elif (use_large_dataset):
        positive_train_name = "train_pos_full.txt"
        negative_train_name = "train_neg_full.txt"

    positive_train_file = os.path.join(data_path, positive_train_name)
    negative_train_file = os.path.join(data_path, negative_train_name)
    assert os.path.isfile(positive_train_file)
    assert os.path.isfile(negative_train_file)
    positive_train_dataframe = _create_dataframe_of_data_and_label(text_file_path=positive_train_file, sentiment=1, parsing_function=parsing_function)
    negative_train_dataframe = _create_dataframe_of_data_and_label(text_file_path=negative_train_file, sentiment=0, parsing_function=parsing_function)
    return positive_train_dataframe.append(negative_train_dataframe)


def _create_dataframe_of_data_and_label(text_file_path: str, sentiment: int, parsing_function=None):
    """
    Takes a text_file_path and generates a dataframe. The input text is parsed with parsing_function (if not None),
    and then put into a dataframe where each row corresponds to one line of the text file. The columns are retrievable
    by the index "text" for the actual text of the corresponding line, and "label" for its sentiment (1 for positive, 0
    for negative.
    :param data_frame_name:
    :param text_file_path:
    :param sentiment:
    :param parsing_function: Function that takes string as input and returns string
    :return: dataframe
    """
    assert os.path.isfile(text_file_path)
    assert isinstance(sentiment, int)
    if (parsing_function != None):
        assert callable(parsing_function)

    with open(text_file_path) as f:
        text_lines = f.read().splitlines()
        if(parsing_function != None):
            parsed_text_lines = []
            for text_line in text_lines:
                parsed_text_lines.append(parsing_function(text_line))
            text_lines = parsed_text_lines

    text_lines_count = len(text_lines)
    text_dataframe = pd.DataFrame(text_lines, columns=["text"])
    if (sentiment):
        label = np.ones(text_lines_count, dtype=np.int8)
    else:
        label = np.zeros(text_lines_count, dtype=np.int8)
    text_dataframe["label"] = label
    assert text_dataframe["label"][0] == sentiment
    return text_dataframe


def parse_test_file(data_path: str, parsing_function=None):
    assert os.path.isdir(data_path)
    if (parsing_function != None):
        assert callable(parsing_function)

    test_file_path = os.path.join(data_path, "test_data.txt")
    assert os.path.isfile(test_file_path)

    # in order to no have to change the whole dataloaders etc. I will add a dummy label to the test dataset.
    test_dataframe = _create_dataframe_of_data_and_label(text_file_path=test_file_path, sentiment=0, parsing_function=parsing_function)
    assert test_dataframe["text"][0][:2] != "1,"
    return test_dataframe


def remove_line_index_and_comma_of_sentence(sentence:str):
    return sentence[2:]