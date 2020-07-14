from modules.bertweet import BERTweet
from torch.utils.data import ConcatDataset, TensorDataset
from utilities.data_loading import generate_bootstrap_dataset


class BERTweetAddData(BERTweet):
    def prepare_data(self) -> None:
        super().prepare_data()

        additional_negative_tweets = self._load_unique_tweets(
            self.config.additional_negative_tweets_path
        )
        additional_positive_tweets = self._load_unique_tweets(
            self.config.additional_positive_tweets_path
        )
        additional_labels = self._generate_labels(
            len(additional_negative_tweets), len(additional_positive_tweets)
        )
        all_additional_tweets = additional_negative_tweets + additional_positive_tweets
        additional_token_id_list = [
            self._encode(self._split_into_tokens(tweet))
            for tweet in all_additional_tweets
        ]

        additional_token_ids = self._pad(
            additional_token_id_list, self.config.max_tokens_per_tweet
        )
        additional_attention_mask = self._generate_attention_mask(additional_token_ids)
        additional_train_data = TensorDataset(
            additional_token_ids, additional_attention_mask, additional_labels,
        )

        self.train_data = ConcatDataset(  # type: ignore
            [self.train_data, additional_train_data]
        )

        if self.config.do_bootstrap_sampling:
            self.train_data = generate_bootstrap_dataset(self.train_data)
