import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer

OUTPUT_DIR = './outputs'


class BERT:

    def __init__(self, layers, neurons, max_seq_length):
        if neurons == 128:
            bert_model_hub = "https://tfhub.dev/tensorflow/bert_uncased_L-{}_H-{}_A-2/1".format(layers, neurons)
        elif neurons == 256:
            bert_model_hub = "https://tfhub.dev/tensorflow/bert_uncased_L-{}_H-{}_A-4/1".format(layers, neurons)
        elif neurons == 512:
            bert_model_hub = "https://tfhub.dev/tensorflow/bert_uncased_L-{}_H-{}_A-8/1".format(layers, neurons)
        elif neurons == 768:
            bert_model_hub = "https://tfhub.dev/tensorflow/bert_en_uncased_L-{}_H-{}_A-12/1".format(layers, neurons)
        else:
            raise ValueError("No BERT model available for {} neurons!".format(neurons))
        self.bert_layer = hub.KerasLayer(bert_model_hub, trainable=True)
        self.MAX_SEQ_LENGTH = max_seq_length

    def create_tokenizer_from_module(self):
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()

        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()

        return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def get_ids(self, tokens, tokenizer, max_seq_length):
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
        return input_ids

    def get_masks(self, tokens, max_seq_length):
        return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))

    def get_segments(self, tokens, max_seq_length):
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))

    def convert_single_example(self, sentence, tokenizer, max_length):

        s_tokens = tokenizer.tokenize(sentence)

        s_tokens = s_tokens[:max_length]

        s_tokens = ["[CLS]"] + s_tokens + ["[SEP]"]

        ids = self.get_ids(s_tokens, tokenizer, self.MAX_SEQ_LENGTH)
        masks = self.get_masks(s_tokens, self.MAX_SEQ_LENGTH)
        segments = self.get_segments(s_tokens, self.MAX_SEQ_LENGTH)

        return ids, masks, segments

    def transform_sentences(self, sentences, tokenizer):
        input_ids, input_masks, input_segments = [], [], []

        for sentence in sentences:
            ids, masks, segments = self.convert_single_example(sentence, tokenizer, self.MAX_SEQ_LENGTH - 2)

            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)

        return [np.asarray(input_ids, dtype=np.int32),
                np.asarray(input_masks, dtype=np.int32),
                np.asarray(input_segments, dtype=np.int32)]

    def create_fc_model(self, dropout_rate, classes):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32,
                                            name="segment_ids")
        _, sequence_outputs = self.bert_layer([input_word_ids, input_mask, segment_ids])
        dense = tf.keras.layers.Dense(64, activation='relu')(sequence_outputs)
        pred = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(dense)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pred)

        return model
