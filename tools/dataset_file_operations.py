import os
import re


def join_two_text_files(
    text_file_1_path: str, text_file_2_path: str, output_file_path: str
):
    assert os.path.isfile(text_file_1_path)
    assert os.path.isfile(text_file_2_path)
    with open(text_file_1_path) as file_1:
        file_1_content = file_1.readlines()
        with open(text_file_2_path) as file_2:
            file_2_content = file_2.readlines()
            joined_file_content = file_1_content + file_2_content
    with open(output_file_path, "w") as output_file:
        output_file.writelines(joined_file_content)


def clean_and_split_test_tweets_to_sepearate_files(
    tweet_file_path: str, output_folder_path: str
):
    assert os.path.isfile(tweet_file_path)
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
    with open(tweet_file_path) as tweet_file:
        tweets = tweet_file.readlines()
        for i, tweet in enumerate(tweets):
            output_file_path = os.path.join(output_folder_path, "{}.txt".format(str(i)))
            with open(output_file_path, "w") as output_file:
                cleaned_tweet = re.sub(r"^.*?,", "", tweet)
                output_file.write(cleaned_tweet)
