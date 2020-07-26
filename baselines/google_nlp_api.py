import os
import time

from comet_ml import Experiment

import numpy as np
import pandas as pd
from data_processing.data_loading_and_storing import load_test_tweets
from google.cloud.language import LanguageServiceClient, enums, types
from google.protobuf.json_format import MessageToDict
from utilities.general_utilities import (
    build_save_path,
    get_args,
    get_bunch_config_from_json,
)


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.set_name(config.experiment_name)
    comet_experiment.log_parameters(config)

    test_tweets = load_test_tweets(config.test_data_path)

    client = LanguageServiceClient()
    result = []
    predictions = np.zeros(len(test_tweets), dtype=np.int32)

    for i, tweet in enumerate(test_tweets):
        start_iter_timestamp = time.time()
        document = types.Document(
            type=enums.Document.Type.PLAIN_TEXT, content=tweet, language="en"
        )

        response = client.analyze_sentiment(document=document)
        response_dict = MessageToDict(response)
        result.append(response_dict)

        prediction_present = bool(response_dict["documentSentiment"])
        if prediction_present:
            # -1, 1 predictions
            predictions[i] = 2 * (response.document_sentiment.score > 0) - 1

        print("iteration", i, "took:", time.time() - start_iter_timestamp, "seconds")

    comet_experiment.log_asset_data(result, name="google_nlp_api_response.json")

    ids = np.arange(1, len(test_tweets) + 1).astype(np.int32)
    predictions_table = np.column_stack((ids, predictions))

    if comet_experiment.disabled:
        save_path = build_save_path(config)
        os.makedirs(save_path)
        pd.DataFrame(
            predictions_table, columns=["Id", "Prediction"], dtype=np.int32,
        ).to_csv(os.path.join(save_path, "google_nlp_api_predictions.csv"), index=False)
    else:
        comet_experiment.log_table(
            filename="google_nlp_api_predictions.csv",
            tabular_data=predictions_table,
            headers=["Id", "Prediction"],
        )

    percentage_predicted = np.sum(predictions != 0) / predictions.shape[0]
    comet_experiment.log_metric(name="percentage predicted", value=percentage_predicted)


if __name__ == "__main__":
    main()
