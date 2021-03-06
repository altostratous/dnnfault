import os

from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_model

from base.experiments import ExperimentBase
from tensorflow import keras
import tensorflow as tf
import logging


logger = logging.getLogger(__name__)


class AlexNet(ExperimentBase):
    checkpoint_filepath = 'tmp/weights/alexnet/alexnet'
    training_epochs = 250

    def get_model(self, name=None):
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                input_shape=(227, 227, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        try:
            model.load_weights(self.checkpoint_filepath)
        except Exception as e:
            logger.error(str(e))
        return model

    def get_configs(self):
        pass

    def evaluate(self, model, x, y_true):
        pass

    def get_dataset(self):
        return keras.datasets.cifar10

    def compile_model(self, model):
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.optimizers.SGD(lr=0.001),
            metrics=['accuracy'])

    def get_first_base_evaluation(self):
        pass

    @staticmethod
    def process_images(image, label):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (227, 227))
        return image, label

    def train(self):
        (train_images, train_labels), (test_images, test_labels) = self.get_dataset().load_data()
        CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_images, train_labels = train_images[5000:], train_labels[5000:]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
        test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
        validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
        print("Training data size:", train_ds_size)
        print("Test data size:", test_ds_size)
        print("Validation data size:", validation_ds_size)

        train_ds = (train_ds
                    .map(self.process_images)
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=8, drop_remainder=True))
        test_ds = (test_ds
                   .map(self.process_images)
                   .shuffle(buffer_size=train_ds_size)
                   .batch(batch_size=8, drop_remainder=True))
        validation_ds = (validation_ds
                         .map(self.process_images)
                         .shuffle(buffer_size=train_ds_size)
                         .batch(batch_size=8, drop_remainder=True))
        model = self.get_model()
        self.compile_model(model)
        model.fit(train_ds,
                  epochs=self.training_epochs,
                  validation_data=validation_ds,
                  validation_freq=1, callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.checkpoint_filepath,
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True)])

    def quantize(self):
        model = self.get_model()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        (train_images, train_labels), (test_images, test_labels) = self.get_dataset().load_data()
        train_images, train_labels = train_images[5000:], train_labels[5000:]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
        # train_ds = (train_ds
        #             .map(self.process_images)
        #             .shuffle(buffer_size=train_ds_size)
        #             .batch(batch_size=8, drop_remainder=True))

        def representative_dataset():
            for data in tf.data.Dataset.from_tensor_slices((train_images, train_labels)).map(
                    self.process_images
            ).batch(1).take(100):
                yield [data.astype(tf.float32)]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_quant_model = converter.convert()