import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=64):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.autotune = tf.data.AUTOTUNE

    def load_data(self, csv_file, image_folder):
        df = pd.read_csv(os.path.join(self.dataset_path, csv_file))
        df["file_name"] = df["file_name"].apply(lambda x: os.path.join(self.dataset_path, image_folder, os.path.basename(x)))
        return df

    def preprocess_image(self, image_path, label=None):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return (image, label) if label is not None else image

    def prepare_dataset(self, df, is_training=True):
        paths, labels = df["file_name"].values, pd.get_dummies(df["label"]).astype(int).values
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(self.preprocess_image, num_parallel_calls=self.autotune)
        if is_training:
            dataset = dataset.shuffle(1000)
        return dataset.batch(self.batch_size).prefetch(self.autotune)
