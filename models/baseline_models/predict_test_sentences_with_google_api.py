import argparse
import os

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


def predict_sentiment_with_google(text_folder_path: str, results_file_path: str):
    """
    Run a google sentiment analysis request on all text files within the given folder.
    :param text_folder_path:
    :return:
    """
    assert os.path.isdir(text_folder_path)
    assert os.path.isfile(results_file_path)
    text_file_names = os.listdir(text_folder_path)

    client = language.LanguageServiceClient()

    for text_file_name in text_file_names:
        with open(os.path.join(text_folder_path, text_file_name), "r") as text_file:
            text_content = text_file.read()

        document = types.Document(
            content=text_content, type=enums.Document.Type.PLAIN_TEXT
        )
        annotations = client.analyze_sentiment(document=document)

        score = annotations.document_sentiment.score
        magnitude = annotations.document_sentiment.magnitude

        with open(results_file_path, "w+") as results_file:
            results_file.write(
                "Textfile: {}, Score: {}, Magnitude: {}".format(
                    text_file_name, score, magnitude
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--text_folder_path",
        help="Path to folder with text files that you want to analyze with google sentiment analysis api",
    )
    args = parser.parse_args()
    predict_sentiment_with_google(args.text_folder_path)
