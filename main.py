from dataloader import DataLoader
from BERT import BERT
import tensorflow as tf


def main():
    dl = DataLoader('twitter-datasets')
    train_df, test_df = dl.load_twitter_dataset(full=False)

    bert = BERT(12, 768, max_seq_length=50)

    BERTtokenizer = bert.create_tokenizer_from_module()

    inputs = bert.transform_sentences(train_df['sentence'].to_numpy(), BERTtokenizer)

    model = bert.create_fc_model(dropout_rate=0.2, classes=2)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    model.fit(x=inputs, y=train_df['polarity'].to_numpy(), epochs=3, batch_size=1)


if __name__ == "__main__":
    main()
