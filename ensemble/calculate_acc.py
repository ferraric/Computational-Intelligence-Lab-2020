import argparse

import numpy as np
from sklearn.metrics import accuracy_score


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i", "--input_file", required=True,
    )
    argparser.add_argument(
        "-l", "--labels", required=True,
    )
    args = argparser.parse_args()

    predictions = np.loadtxt(args.input_file, delimiter=",", skiprows=1, usecols=(1,))
    labels = np.loadtxt(args.labels)

    print("Accuracy:", accuracy_score(labels, predictions))


if __name__ == "__main__":
    main()
