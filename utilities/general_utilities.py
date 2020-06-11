import argparse
import json

from bunch import Bunch


def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c",
        "--config",
        help="Add the Configuration file that has all the relevant parameters",
    )
    return argparser.parse_args()


def get_bunch_config_from_json(json_file_path: str) -> Bunch:
    """
    Get the config from a json file and save it as a Bunch object.
    :param json_file:
    :return: config as Bunch object:
    """
    with open(json_file_path, "r") as config_file:
        config_dict = json.load(config_file)
    return Bunch(config_dict)
