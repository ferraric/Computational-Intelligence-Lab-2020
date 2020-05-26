import argparse
import json
from bunch import Bunch


def get_args() -> argparse:
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-c",
        "--config",
        metavar="C",
        default="None",
        help="Add the Configuration file that has all the relevant parameters",
    )
    args = argparser.parse_args()
    return args


def _get_config_from_json(json_file: str) -> Bunch:
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    return config


def process_config(json_file: str) -> Bunch:
    config = _get_config_from_json(json_file)
    return config
