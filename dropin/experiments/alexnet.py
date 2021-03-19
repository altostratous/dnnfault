import logging
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_apply, \
    quantize_annotate_layer
from base.experiments import ExperimentBase
from dropin.utils import Dropin

logger = logging.getLogger(__name__)


class AlexNet(ExperimentBase):
    checkpoint_filepath = 'tmp/weights/alexnet/alexnet'
    training_epochs = 250
    variants = ExperimentBase.variants + (
        'dropin',
    )
    default_config = {
        'mode': 'evaluation'
    }
    model_name = 'AlexNet'

    def get_model(self, name=None, training_variant='dropin'):
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
        try:
            model.load_weights(self.get_checkpoint_filepath(variant=training_variant))
        except Exception as e:
            logger.error(str(e))
        dropin = Dropin(model, a=0, b=257, r=0.1)
        model = dropin.augment_model(model)
        setattr(model, 'dropin', dropin)
        return model

    def get_configs(self):
        return [{'mode': 'no_fault'}, {'mode': 'evaluation'}]

    def evaluate(self, model, x, y_true, config):
        if config['mode'] == 'evaluation':
            augmented_data = model.dropin.augment_data(x)
        elif config['mode'] == 'no_fault':
            augmented_data = model.dropin.augment_zero(x)
        else:
            assert False
        y_pred = model.predict(x=augmented_data,
                               batch_size=64)
        evaluation = {
            'acc': top_k_categorical_accuracy(y_true, y_pred, k=1),
            'y_true': np.argmax(y_true, axis=1),
            'y_pred': np.argsort(y_pred, axis=1).T[-5:].T
        }
        logger.info('Evaluation Accuracy: {}'.format(np.average(evaluation['acc'])))
        return evaluation

    def get_faulty_model(self, config, name=None):
        return self.get_model(name=name)

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

    def get_first_base_evaluation(self):
        pass

    @staticmethod
    def process_images(image, label=None):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (227, 227))
        return image, label

    def train(self, dropin=False):
        batch_size = 16
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        # validation_images, validation_labels = train_images[:500], train_labels[:500]
        train_images, train_labels = train_images[5000:], train_labels[5000:]
        # train_images, train_labels = train_images[:500], train_labels[:500]

        if dropin:
            model = self.get_model(training_variant='dropin')
        else:
            model = self.get_model()
        self.compile_model(model)

        if dropin:
            data_augmenter = model.dropin.augment_data
        else:
            data_augmenter = model.dropin.augment_zero

        class CIFAR10Sequence(Sequence):

            def __init__(self, x_set, y_set, batch_size, processor, augmenter=data_augmenter):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size
                self.processor = processor
                self.augmenter = augmenter

            def __len__(self):
                return int(np.ceil(len(self.x) / self.batch_size))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                return self.augmenter(self.processor(batch_x)[0]), batch_y

        model.run_eagerly = True
        model.fit(CIFAR10Sequence(train_images, train_labels, batch_size, self.process_images),
                  epochs=self.training_epochs,
                  validation_data=CIFAR10Sequence(validation_images, validation_labels, batch_size,
                                                  self.process_images, augmenter=model.dropin.augment_zero),
                  validation_freq=1, callbacks=[tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.get_checkpoint_filepath(variant='' if not dropin else 'dropin'),
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True)])

    def get_checkpoint_filepath(self, variant=''):
        if variant == 'none':
            variant = ''
        return self.checkpoint_filepath + ('_' if variant else '') + variant

    def quantize(self):
        model = self.get_model()

        def apply_quantization_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
                return quantize_annotate_layer(layer)
            return layer

        # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense`
        # to the layers of the model.
        annotated_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_quantization_to_dense,
        )

        # Now that the Dense layers are annotated,
        # `quantize_apply` actually makes the model quantization aware.
        quant_aware_model = quantize_apply(annotated_model)
        quant_aware_model.summary()
        (train_images, train_labels), (test_images, test_labels) = self.get_dataset().load_data()
        self.compile_model(quant_aware_model)
        self.compile_model(model)
        CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        for x, y in tf.data.Dataset.from_tensor_slices((test_images, test_labels)).map(
                self.process_images
        ).shuffle(buffer_size=1024).take(10).batch(1):
            plt.title(CLASS_NAMES[y[0][0]])
            plt.imshow(x[0])
            print([CLASS_NAMES[i] for i in tf.argsort(quant_aware_model.predict(x))[0]])
            print([CLASS_NAMES[i] for i in tf.argsort(model.predict(x))[0]])
            plt.show()
        exit()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        (train_images, train_labels), (test_images, test_labels) = self.get_dataset().load_data()
        train_images, train_labels = train_images[5000:], train_labels[5000:]

        def representative_dataset():
            for data, label in tf.data.Dataset.from_tensor_slices((train_images, train_labels)).map(
                    self.process_images
            ).batch(1).take(100):
                yield [tf.cast(data, tf.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_quant_model = converter.convert()
        logger.info('Model quantized for tensorflow lite successfully')

    def train_with_dropin(self):
        self.train(dropin=True)

    def get_variant_dropin(self, model, name):
        return self.get_model(name=name, training_variant='dropin')

    def get_variant_none(self, model, name):
        return self.get_model(name=name, training_variant='none')

    def get_plots(self):
        plots = {
            'vulnerable': (self.model_name + ' SDC', 'accuracy', 'vulnerable', 'hist'),
        }
        return plots

    def vulnerable(self):
        # plt.hist([np.average(e['evaluation']['y_true'] == e['evaluation']['y_pred'].T[-1:].T) for e in self.evaluations if e['variant_key'] == 'dropin'])
        plt.hist([np.average(e['evaluation']['y_true'] == e['evaluation']['y_pred'].T[-1:].T) for e in self.evaluations if e['variant_key'] == 'dropin' and e['config']['mode'] == 'no_fault'])
        plt.show()
