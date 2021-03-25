import logging
import os
import pickle

import tensorflow as tf
from tensorflow import keras

from base.utils import insert_layer_nonseq
from clippervsranger.layers import ClipperLayer, ProfileLayer
from dropin.experiments import DropinBase
from dropin.utils import Dropin, CIFAR10Sequence
import numpy as np

logger = logging.getLogger(__name__)


class ClipperSimpleDense(DropinBase):
    checkpoint_filepath = 'tmp/weights/simple_dense/simple_dense'
    training_epochs = 5
    model_name = 'ClipperSimpleDense'

    variants = (
        'dropin',
        'clipper'
    )

    activation_name_pattern = '.*_conv[\d]|.*fc[\d]'

    bounds_file_name = 'dropin/resources/simple_dense_bounds.pkl'
    if os.path.exists(bounds_file_name):
        with open(bounds_file_name, mode='rb') as f:
            bounds = pickle.load(f)

    def profile(self):
        model = self.get_variant_profiler(self.get_model(training_variant='none'))
        self.compile_model(model)
        model.run_eagerly = True
        for x, y in self.get_dataset():
            model.evaluate(model.dropin.augment_zero(x), y)
        bounds = {n: {'upper': max(map(np.max, p)), 'lower': min(map(np.min, p))} for n, p in
                  ProfileLayer.profile.items()}
        with open(self.bounds_file_name, mode='wb') as f:
            pickle.dump(bounds, f)
        print(bounds)

    def get_variant_profiler(self, faulty_model, name=None):
        model = self.copy_model(faulty_model, name=(name or '') + '_base_copy')

        def ranger_layer_factory(insert_layer_name):
            return ProfileLayer(name=insert_layer_name)

        model = insert_layer_nonseq(model, self.activation_name_pattern, ranger_layer_factory, 'dummy', model_name=name)
        setattr(model, 'dropin', faulty_model.dropin)
        return model

    def get_model(self, name=None, training_variant='dropin'):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, input_shape=(784,), activation='relu'),
            tf.keras.layers.Dense(10, input_shape=(128,), activation='softmax')
        ], name=name)

        try:
            model.load_weights(self.get_checkpoint_filepath(variant=training_variant))
        except Exception as e:
            logger.error(str(e))
        if training_variant == 'dropin':
            dropin = Dropin(model, a=0, b=19.840586, r=0.1)
        else:
            dropin = Dropin(model, a=0, b=16.049778, r=0.1)
        model = dropin.augment_model(model)
        setattr(model, 'dropin', dropin)
        return model

    def get_dataset(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        s = tf.data.Dataset.cardinality(test_ds)
        test_ds = test_ds.shuffle(buffer_size=s).batch(256)
        return test_ds

    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.run_eagerly = True

    def train(self, dropin=False):
        batch_size = 32
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        if dropin:
            model = self.get_model(training_variant='dropin')
        else:
            model = self.get_model()
        self.compile_model(model)

        if dropin:
            data_augmenter = model.dropin.augment_data
        else:
            data_augmenter = model.dropin.augment_zero

        model.run_eagerly = True
        model.fit(CIFAR10Sequence(x_train, y_train, batch_size, augmenter=data_augmenter),
                  epochs=self.training_epochs,
                  validation_data=CIFAR10Sequence(x_test, y_test, batch_size, augmenter=data_augmenter),
                  validation_freq=1, callbacks=[tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.get_checkpoint_filepath(variant='' if not dropin else 'dropin'),
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True)])

    def get_profile_database(self, dropin_model):
        batch_size = 32
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return CIFAR10Sequence(x_test, y_test, batch_size, augmenter=dropin_model.dropin.augment_zero)

    def get_variant_clipper(self, faulty_model, name=None):
        model = self.get_model(name=name, training_variant='none')

        def ranger_layer_factory(insert_layer_name):
            return ClipperLayer(name=insert_layer_name, bounds=self.bounds)
        model = insert_layer_nonseq(model, self.activation_name_pattern, ranger_layer_factory, 'dummy', model_name=name)
        return model