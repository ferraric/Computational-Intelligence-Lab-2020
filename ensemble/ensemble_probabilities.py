import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def get_model_probabilty_files(input_folder: str) -> List[str]:
    model_probabilitiy_files = []
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if is_valid_input_file(file_path):
            model_probabilitiy_files.append(file_path)
    return model_probabilitiy_files


def is_valid_input_file(file_name: str) -> bool:
    return (
        os.path.isfile(file_name)
        and file_name.endswith(".csv")
        and "probabilities" in file_name
    )


def get_probabilities(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float, skiprows=1, usecols=(1,),)


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        help="Contains the output prediction probabilities from models to ensemble",
    )
    argparser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Path, to which where ensemble prediciton is saved",
    )
    args = argparser.parse_args()

    model_probabilitiy_files = get_model_probabilty_files(args.input_folder)
    model_probabilities = np.array(
        [get_probabilities(f) for f in model_probabilitiy_files]
    )

    avg_model_probabilities = np.mean(model_probabilities, axis=0)
    ensemble_predictions = (2 * (avg_model_probabilities > 0.5) - 1).astype(np.int32)
    ids = np.arange(1, ensemble_predictions.shape[0] + 1).astype(np.int32)

    prediction_table = pd.DataFrame(
        np.stack([ids, ensemble_predictions], axis=1), columns=["Id", "Prediction"]
    )
    prediction_table.to_csv(
        os.path.join(args.output_path, "ensemble_predictions.csv"), index=False
    )


if __name__ == "__main__":
    main()
