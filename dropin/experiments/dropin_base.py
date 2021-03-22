from abc import ABC, ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from tensorflow import keras
from tensorflow.python.keras.metrics import sparse_top_k_categorical_accuracy
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_layer, \
    quantize_apply

from base.experiments import ExperimentBase
from dropin.utils import Dropin, CIFAR10Sequence

import logging

logger = logging.getLogger(__name__)


class DropinBase(ExperimentBase, ABC, metaclass=ABCMeta):
    checkpoint_filepath = None
    training_epochs = 250
    variants = ExperimentBase.variants + (
        'dropin',
    )
    default_config = {
        'mode': 'evaluation'
    }
    model_name = None

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
            'acc': sparse_top_k_categorical_accuracy(y_true, y_pred, k=1),
            'y_true': y_true,
            'y_pred': self.get_top_k(y_pred)
        }
        logger.info('Evaluation Accuracy: {}'.format(np.average(evaluation['acc'])))
        return evaluation

    def get_top_k(self, y_pred, k=5):
        return np.argsort(y_pred, axis=1).T[-k:].T

    def get_faulty_model(self, config, name=None):
        return self.get_model(name=name)

    def get_first_base_evaluation(self):
        pass

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

    def get_variant_dropin(self, model, name=None):
        return self.get_model(name=name, training_variant='dropin')

    def get_variant_none(self, model, name=None):
        return self.get_model(name=name, training_variant='none')

    def get_plots(self):
        plots = {
            'vulnerable': (self.model_name + ' SDC', 'accuracy', 'vulnerable', 'hist'),
        }
        return plots

    def vulnerable(self):
        for variant, condition in (
            ('none-fault', lambda e: e['variant_key'] == 'none' and e['config']['mode'] == 'evaluation'),
            ('none-no-fault', lambda e: e['variant_key'] == 'none' and e['config']['mode'] == 'no_fault'),
            ('dropin-fault', lambda e: e['variant_key'] == 'dropin' and e['config']['mode'] == 'evaluation'),
            ('dropin-no-fault', lambda e: e['variant_key'] == 'dropin' and e['config']['mode'] == 'no_fault'),
        ):
            evaluation = [
                np.average(e['evaluation']['acc'])
                for e in self.evaluations if condition(e)]
            probability = []
            x = []
            for step in range(0, 1001, 5):
                percentage = step / 1000
                x.append(percentage)
                probability.append(len([i for i in evaluation if i < percentage]) / len(evaluation))
            plt.plot(x, probability, label=variant)
        plt.xlabel('desired degraded accuracy')
        plt.ylabel('portion of vulnerable parameters')
        plt.title('Parameters Vulnerable Portion VS Desired Degraded Accuracy')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
        plt.legend()
        plt.show()

    def profile_dropin(self):
        dropin_model = self.get_variant_dropin(None)
        d = Dropin(dropin_model, representative_dataset=self.get_profile_database(dropin_model))
        # dropin 0.0, 972.57635
        # none 0.0 953.709

        print(d.a, d.b)

    @abstractmethod
    def get_profile_database(self, dropin_model):
        pass

    @abstractmethod
    def train(self, dropin=False):
        pass
