import pandas as pd
from sklearn.model_selection import train_test_split

from utils.dataset_loader import DatasetLoader
from utils.model import EfficientNetModel
from utils.trainer import ModelTrainer
from utils.predictor import ModelPredictor


if __name__ == "__main__":
    # Define paths
    dataset_path = "data"
    train_csv = "train.csv"
    test_csv = "test.csv"
    
    # Load dataset
    data_loader = DatasetLoader(dataset_path,  img_size=(224, 224), batch_size=16)
    train_df = data_loader.load_data(train_csv, "train_data")
    test_df = data_loader.load_data(test_csv, "test_data_v2")

    # Split data
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

    # Prepare datasets
    train_dataset = data_loader.prepare_dataset(train_df, is_training=True)
    val_dataset = data_loader.prepare_dataset(val_df, is_training=False)

    # Create model
    model_builder = EfficientNetModel(num_classes=train_df['label'].nunique())
    model = model_builder.model
    model_builder.compile_model()

    # Train model
    trainer = ModelTrainer(model, train_dataset, val_dataset)
    # trainer.train(epochs=20)

    # Load trained model and evaluate
    predictor = ModelPredictor("model/best_model.keras")
    predictions = predictor.predict(val_dataset)
    print(predictions)
    predictor.evaluate(pd.get_dummies(val_df["label"]).values, predictions)
    predictor.plot_roc_curve(pd.get_dummies(val_df["label"]).values, predictions)
