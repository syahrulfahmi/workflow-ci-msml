import mlflow
import pandas as pd
import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


if __name__ == "__main__":
    # load data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "ulasan_processed_dataset.csv")
    df = pd.read_csv(csv_path)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['polarity'])
    num_classes = len(le.classes_)

    X = df['text_akhir'].astype(str).values
    y = df['label'].values

    # split data train dan data test
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # text tokenizer
    max_words = 30000
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_lstm)

    # parameter
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    max_len = 100
    embedding_dim = 128

    X_train_seq = tokenizer.texts_to_sequences(X_train_lstm)
    X_test_seq = tokenizer.texts_to_sequences(X_test_lstm)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    print("Vocab size:", vocab_size)


    def build_lstm_model():
        model_sequential = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            Bidirectional(LSTM(128, return_sequences=False)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model_sequential.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=["accuracy"]
        )
        model_sequential.build(input_shape=(None, max_len))
        return model_sequential


    model = build_lstm_model()
    model.summary()

    es = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    input_example = X_train_pad[:5]

    with mlflow.start_run():
        mlflow.autolog()

        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("lstm_units", 128)
        mlflow.log_param("max_words", max_words)
        mlflow.log_param("max_len", max_len)
        mlflow.log_param("num_classes", num_classes)

        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            input_example=input_example
        )

        model.fit(
            X_train_pad, y_train_lstm,
            validation_split=0.1,
            epochs=20,
            batch_size=128,
            callbacks=[es],
            verbose=1
        )

    print("\nSelesai Train Model")
