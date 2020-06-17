import tensorflow as tf
from BERT import BERT
import numpy as np
import pandas as pd
from dataloader import DataLoader

unsup_ratio = 0.05
num_samples = 220000

def kl_for_log_probs(log_p, log_q):
    p = tf.exp(log_p)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl

def uda_loss(y_true, y_pred):
    # assert num_samples % (1 + 2 * unsup_ratio) == 0
    sup_batch_size = int(num_samples // (1 + 2 * unsup_ratio))
    unsup_batch_size = int(sup_batch_size * unsup_ratio)

    sup_log_probs = y_pred[:sup_batch_size]
    # one_hot_labels = tf.one_hot(indices=y_true[:sup_batch_size], depth=2)
    # target_one_hot = one_hot_labels

    per_example_loss = tf.reduce_sum(y_true[:sup_batch_size] * sup_log_probs, axis=-1)
    loss_mask = tf.ones_like(per_example_loss)

    per_example_loss = per_example_loss * loss_mask
    sup_loss = (tf.reduce_sum(per_example_loss) /
                tf.maximum(tf.reduce_sum(loss_mask), 1))

    orig_start = sup_batch_size
    orig_end = orig_start + unsup_batch_size
    aug_start = sup_batch_size + unsup_batch_size
    aug_end = aug_start + unsup_batch_size

    ori_log_probs = y_pred[orig_start:orig_end]
    aug_log_probs = y_pred[aug_start:aug_end]
    per_example_kl_loss = kl_for_log_probs(
        ori_log_probs, aug_log_probs)

    unsup_loss = tf.reduce_mean(per_example_kl_loss)

    return sup_loss + unsup_loss

def run_uda():

    bert = BERT()
    dl = DataLoader('twitter-datasets')
    train_df, test_df, test_bt_df = dl.load_twitter_dataset(full=False)

    tokenizer = bert.create_tokenizer_from_module()
    inputs = bert.transform_sentences(train_df['sentence'].to_numpy(), tokenizer)
    test_inputs = bert.transform_sentences(test_df['sentence'].to_numpy(), tokenizer)
    test_bt_inputs = bert.transform_sentences(test_bt_df['sentence'].to_numpy(dtype=np.str), tokenizer)

    sup_input_ids = inputs[0]
    unsup_orig_input_ids = test_inputs[0]
    unsup_aug_input_ids = test_bt_inputs[0]

    sup_input_mask = inputs[1]
    unsup_orig_input_mask = test_inputs[1]
    unsup_aug_input_mask = test_bt_inputs[1]

    sup_input_segments = inputs[2]
    unsup_orig_input_segments = test_inputs[2]
    unsup_aug_input_segments = test_bt_inputs[2]

    input_ids = tf.concat([
        sup_input_ids,
        unsup_orig_input_ids,
        unsup_aug_input_ids
    ], 0)

    input_mask = tf.concat([
        sup_input_mask,
        unsup_orig_input_mask,
        unsup_aug_input_mask
    ], 0)

    input_segments = tf.concat([
        sup_input_segments,
        unsup_orig_input_segments,
        unsup_aug_input_segments
    ], 0)

    unsup_samples = unsup_orig_input_ids.shape[0]
    one_hot_labels = tf.keras.utils.to_categorical(tf.concat([train_df['polarity'].to_numpy(), np.ones(2*unsup_samples)], 0), num_classes=2)
    one_hot_labels.astype(np.int64, copy=False)

    model = bert.create_fc_model(dropout_rate=0.1)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-5), loss=uda_loss, metrics=['accuracy'])

    print(model.summary())

    model.fit(x=[input_ids, input_mask, input_segments], y=one_hot_labels, epochs=2, batch_size=16)

    test_inputs = bert.transform_sentences(test_df['sentence'].to_numpy(), tokenizer)

    predicted_labels = np.argmax(model.predict(x=test_inputs), axis=1)
    predicted_labels[predicted_labels == 0] = -1

    result = pd.DataFrame(data=predicted_labels)
    result.index += 1
    result.to_csv('./outputs/results.csv', index_label='Id', header=['Prediction'])