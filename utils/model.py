import tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

class EfficientNetModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=self.input_shape)
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        return model

    def compile_model(self, learning_rate=0.001):
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.metrics import Precision, Recall, AUC

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]
        )
