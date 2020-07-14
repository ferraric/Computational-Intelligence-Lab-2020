from modules.roberta_sentiment_classifier import RobertaSentimentClassifier
from torch.utils.data import ConcatDataset, TensorDataset
from utilities.data_loading import generate_bootstrap_dataset


class RobertaSentimentClassifierAddData(RobertaSentimentClassifier):
    def prepare_data(self) -> None:
        super().prepare_data()
        additional_positive_tweets = self._load_unique_tweets(
            self.config.additional_positive_tweets_path
        )
        additional_negative_tweets = self._load_unique_tweets(
            self.config.additional_negative_tweets_path
        )
        additional_labels = self._generate_labels(
            len(additional_negative_tweets), len(additional_positive_tweets)
        )
        (
            additional_train_token_ids,
            additional_train_attention_mask,
        ) = self._tokenize_tweets(
            self.tokenizer, additional_negative_tweets + additional_positive_tweets
        )
        additional_train_data = TensorDataset(
            additional_train_token_ids,
            additional_train_attention_mask,
            additional_labels,
        )

        self.train_data = ConcatDataset([self.train_data, additional_train_data])  # type: ignore

        if self.config.do_bootstrap_sampling:
            self.train_data = generate_bootstrap_dataset(self.train_data)
