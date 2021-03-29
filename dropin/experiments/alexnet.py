import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Model

from dropin.experiments import DropinBase
from dropin.utils import Dropin, CIFAR10Sequence

logger = logging.getLogger(__name__)


class AlexNetV2(DropinBase):
    checkpoint_filepath = 'tmp/weights/alexnet/alexnet'
    training_epochs = 250
    model_name = 'AlexNet'

    def get_dataset(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        s = tf.data.Dataset.cardinality(test_ds)
        test_ds = test_ds.shuffle(buffer_size=s).batch(256).map(self.process_images)
        return test_ds

    def compile_model(self, model):
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.optimizers.SGD(lr=0.001),
            metrics=['accuracy'])
        model.run_eagerly = True

    @staticmethod
    def process_images(image, label=None):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (227, 227))
        return image, label

    def train(self, training_variant='none'):
        dropin = training_variant != 'none'
        batch_size = 16
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        # validation_images, validation_labels = train_images[:500], train_labels[:500]
        train_images, train_labels = train_images[5000:], train_labels[5000:]
        # train_images, train_labels = train_images[:500], train_labels[:500]

        model = self.get_model(training_variant=training_variant)
        self.compile_model(model)

        if dropin:
            data_augmenter = model.dropin.augment_data
        else:
            data_augmenter = model.dropin.augment_zero

        model.run_eagerly = True
        model.fit(CIFAR10Sequence(train_images, train_labels, batch_size, self.process_images,
                                  data_augmenter),
                  epochs=self.training_epochs,
                  validation_data=CIFAR10Sequence(validation_images, validation_labels, batch_size,
                                                  self.process_images, augmenter=data_augmenter),
                  validation_freq=1, callbacks=[tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.get_checkpoint_filepath(variant='' if not dropin else training_variant),
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True)])

    def get_profile_database(self, dropin_model):
        batch_size = 16
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        return CIFAR10Sequence(validation_images, validation_labels, batch_size,
                               self.process_images, augmenter=dropin_model.dropin.augment_zero)

    def get_raw_model(self, name=None) -> Model:
        model = keras.models.Sequential([
            keras.layers.Layer(input_shape=(227, 227, 3)),
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
        ], name=name)
        default_dropin = self.get_default_dropin(model)
        model = default_dropin.augment_model(model)
        setattr(model, 'dropin', default_dropin)
        return model

    def get_default_dropin(self, model):
        if self.args.tag and self.args.tag == 'worst':
            Dropin(model, r=0.5, mode='worst', a=0, b=2)
        return Dropin(model, r=0.5, mode='random', a=0, b=2)


class AlexNetRandomSmoothing(AlexNetV2):

    variants = DropinBase.variants + (
        'random_smoothing',
    )
    default_config = {
        'mode': 'evaluation'
    }
    checkpoint_filepath = 'tmp/weights/alexnet/alexnet'
    training_epochs = 250
    model_name = 'AlexNetRS'

    def get_default_dropin(self, model):
        if self.args.tag and self.args.tag == 'worst':
            Dropin(model, r=0.5, mode='worst', a=0, b=2)
        return Dropin(model, r=0.5, mode='zero', a=0, b=2, regex='batch_normalization.*',
                      perturb=lambda x, p: x * p)
