from dataloader import DataLoader
from BERT import BERT
import pandas as pd
import numpy as np
import tensorflow as tf


def main():
    dl = DataLoader('twitter-datasets')
    train_df, test_df = dl.load_twitter_dataset(full=False)

    bert = BERT(bert_model_size='bert_base', max_seq_length=50)

    tokenizer = bert.create_tokenizer_from_module()

    inputs = bert.transform_sentences(train_df['sentence'].to_numpy(), tokenizer)
    one_hot_labels = tf.keras.utils.to_categorical(train_df['polarity'].to_numpy(), num_classes=2)

    model = bert.create_fc_model(dropout_rate=0.1)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-5), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    model.fit(x=inputs, y=one_hot_labels, epochs=3, batch_size=32, validation_split=0.2)

    test_inputs = bert.transform_sentences(test_df['sentence'].to_numpy(), tokenizer)

    predicted_labels = model.predict(x=test_inputs)
    np.save('./outputs/results.npy', predicted_labels, allow_pickle=True)

    result = pd.DataFrame(data=np.argmax(predicted_labels, axis=1))
    result.index += 1
    result.to_csv('./outputs/results.csv', index_label='Id', header=['Prediction'])


if __name__ == "__main__":
    main()
