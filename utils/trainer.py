import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, model_path="model/best_model.keras"):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_path = model_path
        self.callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor="val_accuracy", mode="max")
        ]

    def train(self, epochs=20):
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=self.callbacks
        )
        return history

    def evaluate(self):
        results = self.model.evaluate(self.val_dataset)
        print(f"Validation Results: {results}")
