import numpy as np
from comet_ml import Experiment
from google.cloud import language
from google.cloud.language import enums, types
from google.protobuf.json_format import MessageToDict
from utilities.data_loading import load_tweets, remove_indices_from_test_tweets
from utilities.general_utilities import get_args, get_bunch_config_from_json


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.set_name(config.comet_experiment_name)
    comet_experiment.log_parameters(config)

    test_tweets = load_tweets(config.test_data_path)
    test_tweets_index_removed = remove_indices_from_test_tweets(test_tweets)

    client = language.LanguageServiceClient()
    result = []
    predictions = np.zeros(len(test_tweets_index_removed))

    for i, tweet in enumerate(test_tweets_index_removed):
        document = types.Document(
            type=enums.Document.Type.PLAIN_TEXT, content=tweet, language="en"
        )

        response = client.analyze_sentiment(document=document)
        response_dict = MessageToDict(response)
        result.append(response_dict)

        prediction_present = bool(response_dict["documentSentiment"])
        if prediction_present:
            # -1, 1 predictions
            predictions[i] = 2 * response.document_sentiment.score - 1


if __name__ == "__main__":
    main()
