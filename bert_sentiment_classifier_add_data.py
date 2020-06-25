from bert_sentiment_classifier import BertSentimentClassifier
from torch.utils.data import ConcatDataset, TensorDataset
from transformers import BertTokenizerFast


class BertSentimentClassifierAddData(BertSentimentClassifier):
    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        super().prepare_data()

        additional_positive_tweets = self._load_tweets(
            self.config.additional_positive_tweets_path
        )
        additional_negative_tweets = self._load_tweets(
            self.config.additional_negative_tweets_path
        )
        additional_labels = self._generate_labels(
            len(additional_negative_tweets), len(additional_positive_tweets)
        )
        (
            additional_train_token_ids,
            additional_train_attention_mask,
        ) = self._tokenize_tweets(
            tokenizer, additional_negative_tweets + additional_positive_tweets
        )
        additional_train_data = TensorDataset(
            additional_train_token_ids,
            additional_train_attention_mask,
            additional_labels,
        )

        self.train_data = ConcatDataset([self.train_data, additional_train_data])  # type: ignore
