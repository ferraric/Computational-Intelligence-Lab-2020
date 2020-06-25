from bert_sentiment_classifier import BertSentimentClassifier as BSC
from torch.utils.data import ChainDataset, TensorDataset
from transformers import BertTokenizerFast


class BertSentimentClassifierAug(BSC):
    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        negative_tweets = BSC.load_tweets(self, self.config.negative_tweets_path)
        positive_tweets = BSC.load_tweets(self, self.config.positive_tweets_path)

        labels = BSC.generate_labels(self, len(negative_tweets), len(positive_tweets))
        train_token_ids, train_attention_mask = BSC.tokenize_tweets(
            self, tokenizer, negative_tweets + positive_tweets
        )
        self.train_data, self.validation_data = BSC.train_validation_split(
            self,
            self.config.validation_size,
            TensorDataset(train_token_ids, train_attention_mask, labels),
        )
        positive_tweets_aug = BSC.load_tweets(
            self, self.config.augmented_positive_tweets_path
        )
        negative_tweets_aug = BSC.load_tweets(
            self, self.config.augmented_negative_tweets_path
        )
        labels_aug = BSC.generate_labels(
            self, len(negative_tweets_aug), len(positive_tweets_aug)
        )
        train_aug_token_ids, train_aug_attention_mask = BSC.tokenize_tweets(
            self, tokenizer, negative_tweets_aug + positive_tweets_aug
        )
        train_data_aug = TensorDataset(
            train_aug_token_ids, train_aug_attention_mask, labels_aug
        )
        self.train_data = ChainDataset(self.train_data, train_data_aug)

        test_tweets = BSC.load_tweets(self, self.config.test_tweets_path)
        test_token_ids, test_attention_mask = BSC.tokenize_tweets(
            self, tokenizer, test_tweets
        )
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)

        BSC.get_max_sequence_lenth(self, train_attention_mask, test_attention_mask)
