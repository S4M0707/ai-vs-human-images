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

    def _load_and_preprocess(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image
    
    def preprocess_image_with_label(self, path, label):
        image = self._load_and_preprocess(path)
        return image, label

    def preprocess_image_without_label(self, path):
        image = self._load_and_preprocess(path)
        return image
    
    def prepare_dataset(self, df, is_training=True):
        paths = df["file_name"].values

        if is_training:
            labels = pd.get_dummies(df["label"]).astype(int).values
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
            dataset = dataset.map(self.preprocess_image_with_label, num_parallel_calls=self.autotune)
            dataset = dataset.shuffle(1000)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(paths)
            dataset = dataset.map(self.preprocess_image_without_label, num_parallel_calls=self.autotune)

        return dataset.batch(self.batch_size).prefetch(self.autotune)
    
    def prepare_image_paths(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(self.preprocess_image_without_label, num_parallel_calls=self.autotune)
        return dataset.batch(self.batch_size).prefetch(self.autotune)
