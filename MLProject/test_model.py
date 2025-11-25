import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime


if __name__ == "__main__":
    # ======================
    # Buat dataset dummy
    # ======================
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                               n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ======================
    # Definisikan model
    # ======================
    def build_model(input_dim, num_classes):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    model = build_model(input_dim=X_train.shape[1], num_classes=len(np.unique(y)))

    # ======================
    # Setup MLflow
    # ======================

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dl_test_run_{now}"

    input_example = X_train[:5]

    # ======================
    # Training & Logging
    # ======================
    with mlflow.start_run(run_name=run_name):
        mlflow.tensorflow.autolog()

        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=5,
            batch_size=32,
            verbose=1
        )

        # Log model explicitly (opsional karena autolog sudah log model)
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            input_example=input_example
        )

print("Selesai training dan logging model!")
