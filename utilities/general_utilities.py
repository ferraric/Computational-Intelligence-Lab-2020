import argparse
import json
from bunch import Bunch


def get_args() -> argparse:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c",
        "--config",
        help="Add the Configuration file that has all the relevant parameters",
    )
    return argparser.parse_args()


def get_bunch_config_from_json(json_file: str) -> Bunch:
    """
    Get the config from a json file and save it as a Bunch namespace object.
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)
    return Bunch(config_dict)
