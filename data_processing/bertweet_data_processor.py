from typing import List

from bunch import Bunch
from data_processing.data_loading_and_storing import load_test_tweets, load_tweets
from data_processing.data_processor import DataProcessor
from data_processing.tokenizer import Tokenizer


class BertweetDataProcessor(DataProcessor):
    def __init__(self, config: Bunch, tokenizer: Tokenizer):
        super().__init__(config, tokenizer)

    def _replace_special_tokens(self, tweet: str) -> str:
        return tweet.replace("<url>", "HTTPURL").replace("<user>", "@USER")

    def _load_tweets(self, path: str) -> List[str]:
        tweets = load_tweets(path)
        self.logger.experiment.log_other(key="n_tweets_from:" + path, value=len(tweets))
        if self.config.use_special_tokens:
            return [self._replace_special_tokens(tweet) for tweet in tweets]
        else:
            return tweets

    def _load_test_tweets(self, path: str) -> List[str]:
        tweets = load_test_tweets(path)
        self.logger.experiment.log_other(key="n_tweets_from:" + path, value=len(tweets))
        if self.config.use_special_tokens:
            return [self._replace_special_tokens(tweet) for tweet in tweets]
        else:
            return tweets
