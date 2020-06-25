from bert_sentiment_classifier import BertSentimentClassifier
from torch.utils.data import ConcatDataset, TensorDataset
from transformers import BertTokenizerFast


class BertSentimentClassifierAug(BertSentimentClassifier):
    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        BertSentimentClassifier.prepare_data(self)

        additional_positive_tweets = BertSentimentClassifier._load_tweets(
            self, self.config.additional_positive_tweets_path
        )
        additional_negative_tweets = BertSentimentClassifier._load_tweets(
            self, self.config.additional_negative_tweets_path
        )
        additional_labels = BertSentimentClassifier._generate_labels(
            self, len(additional_negative_tweets), len(additional_positive_tweets)
        )
        (
            additional_train_token_ids,
            additional_train_attention_mask,
        ) = BertSentimentClassifier._tokenize_tweets(
            self, tokenizer, additional_negative_tweets + additional_positive_tweets
        )
        additional_train_data = TensorDataset(
            additional_train_token_ids,
            additional_train_attention_mask,
            additional_labels,
        )

        self.train_data = ConcatDataset([self.train_data, additional_train_data])  # type: ignore
