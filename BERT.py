import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from bert import albert_tokenization
from bert import bert_tokenization

OUTPUT_DIR = './outputs'


class BERT:

    def __init__(self, bert_model_size='bert_base', max_seq_length=140):
        if bert_model_size == 'bert_base':
            model_hub = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
            self.tokenizer_type = 'bert'
        elif bert_model_size == 'bert_large':
            model_hub = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2'
            self.tokenizer_type = 'bert'
        elif bert_model_size == 'albert_base':
            model_hub = 'https://tfhub.dev/tensorflow/albert_en_base/1'
            self.tokenizer_type = 'albert'
        elif bert_model_size == 'albert_large':
            model_hub = 'https://tfhub.dev/tensorflow/albert_en_large/1'
            self.tokenizer_type = 'albert'
        else:
            raise ValueError("No model available!")
        self.bert_layer = hub.KerasLayer(model_hub, trainable=True)
        self.MAX_SEQ_LENGTH = max_seq_length

    def create_tokenizer_from_module(self):
        if self.tokenizer_type == 'bert':
            do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
            vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
            return bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        elif self.tokenizer_type == 'albert':
            sp_model_file = self.bert_layer.resolved_object.sp_model_file.asset_path.numpy()
            return albert_tokenization.FullTokenizer(vocab_file=None, spm_model_file=sp_model_file)

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

    def create_fc_model(self, dropout_rate):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32,
                                        name="input_word_ids")
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32,
                                    name="input_mask")
        segment_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32,
                                     name="segment_ids")
        pooled_outputs, sequence_outputs = self.bert_layer([input_word_ids, input_mask, segment_ids])
        logits = tf.keras.layers.Dropout(dropout_rate)(pooled_outputs)
        prob = tf.keras.layers.Dense(2, activation='softmax')(logits)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=prob)

        return model

    # def create_fc_model_for_uda(self,
    #                             input_word_ids,
    #                             input_mask,
    #                             input_segment_ids,
    #                             unsup_ratio,
    #                             dropout_rate):
    #
    #     num_samples = input_word_ids.shape[0]
    #     assert num_samples % (1 + 2 * unsup_ratio) == 0
    #     sup_batch_size = num_samples // (1 + 2 * unsup_ratio)
    #     unsup_batch_size = sup_batch_size * unsup_ratio
    #
    #     pooled_outputs, sequence_outputs = self.bert_layer([input_word_ids, input_mask, input_segment_ids])
    #     logits = tf.keras.layers.Dropout(dropout_rate)(pooled_outputs)
    #     log_probs = tf.keras.layers.Dense(2, activation='softmax')(logits)
    #
    #     def uda_loss(y_true, y_pred):
    #         sup_log_probs = y_pred[:sup_batch_size]
    #         one_hot_labels = tf.one_hot(indices=y_true, depth=2, on_value=1, off_value=-1)
    #         target_one_hot = one_hot_labels
    #
    #         per_example_loss = tf.reduce_sum(target_one_hot * sup_log_probs, axis=-1)
    #         loss_mask = tf.ones_like(per_example_loss)
    #
    #         loss_mask = tf.stop_gradient(loss_mask)
    #         per_example_loss = per_example_loss * loss_mask
    #         sup_loss = (tf.reduce_sum(per_example_loss) /
    #                     tf.maximum(tf.reduce_sum(loss_mask), 1))
    #
    #
    #         orig_start = sup_batch_size
    #         orig_end = orig_start + unsup_batch_size
    #         aug_start = sup_batch_size + unsup_batch_size
    #         aug_end = aug_start + unsup_batch_size
    #
    #         ori_log_probs = y_pred[orig_start:orig_end]
    #         aug_log_probs = y_pred[aug_start:aug_end]
    #         per_example_kl_loss = self.kl_for_log_probs(
    #             ori_log_probs, aug_log_probs)
    #
    #         unsup_loss = tf.reduce_mean(per_example_kl_loss)
    #
    #         return sup_loss + unsup_loss
    #
    #     model = tf.keras.models.Model()
