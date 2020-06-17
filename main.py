from dataloader import DataLoader
from BERT import BERT
import pandas as pd
import numpy as np
import tensorflow as tf
import string
from comet_ml import Experiment
import uda

def main():
    # experiment = Experiment(api_key="DLxzM6ydqoaaoVKuemHt82NLS",
    #                         project_name="sentiment-analysis", workspace="jerry-crea")
    #
    # dl = DataLoader('twitter-datasets')
    # train_df, test_df, _ = dl.load_twitter_dataset(full=False)
    # train_df['sentence'] = train_df['sentence'].apply(lambda text: "".join([c for c in text if c not in string.punctuation]))
    # test_df['sentence'] = test_df['sentence'].apply(lambda text: "".join([c for c in text if c not in string.punctuation]))
    #
    # train_df['sentence'] = train_df['sentence'].apply(lambda text: text.replace('<user>', '').replace('<url>', ''))
    # test_df['sentence'] = test_df['sentence'].apply(lambda text: text.replace('<user>', '').replace('<url>', ''))
    # bert = BERT(bert_model_size='albert_base')
    #
    # tokenizer = bert.create_tokenizer_from_module()
    # inputs = bert.transform_sentences(train_df['sentence'].to_numpy(), tokenizer)
    # one_hot_labels = tf.keras.utils.to_categorical(train_df['polarity'].to_numpy(), num_classes=2)
    #
    # model = bert.create_fc_model(dropout_rate=0.2)
    #
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-5), loss='binary_crossentropy', metrics=['accuracy'])
    #
    # print(model.summary())
    #
    # model.fit(x=inputs, y=one_hot_labels, epochs=2, batch_size=32, validation_split=0.2)
    #
    # test_inputs = bert.transform_sentences(test_df['sentence'].to_numpy(), tokenizer)
    #
    # predicted_labels = np.argmax(model.predict(x=test_inputs), axis=1)
    # predicted_labels[predicted_labels == 0] = -1
    #
    # result = pd.DataFrame(data=predicted_labels)
    # result.index += 1
    # result.to_csv('./outputs/results.csv', index_label='Id', header=['Prediction'])
    uda.run_uda()


if __name__ == "__main__":
    main()
