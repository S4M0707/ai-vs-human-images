from sklearn.metrics import accuracy_score, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class ModelPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, dataset):
        predictions = self.model.predict(dataset)
        return np.argmax(predictions, axis=1)

    def evaluate(self, true_labels, predictions):
        accuracy = accuracy_score(np.argmax(true_labels, axis=1), predictions)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def plot_roc_curve(self, true_labels, predictions):
        fpr, tpr, _ = roc_curve(np.argmax(true_labels, axis=1), predictions)
        plt.plot(fpr, tpr, label="ROC Curve", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
