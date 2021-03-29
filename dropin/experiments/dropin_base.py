import os
import pickle
from abc import ABC, ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from tensorflow.python.keras.metrics import sparse_top_k_categorical_accuracy
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_layer, \
    quantize_apply

from base.experiments import ExperimentBase
from dropin.utils import Dropin

import logging

logger = logging.getLogger(__name__)


class DropinBase(ExperimentBase, ABC, metaclass=ABCMeta):
    checkpoint_filepath = None
    training_epochs = 100
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
        self.train(training_variant='dropin')

    def train_with_random_smoothing(self):
        self.train(training_variant='random_smoothing')

    def get_variant_dropin(self, model, name=None):
        return self.get_model(name=name, training_variant='dropin')

    def get_variant_random_smoothing(self, model, name=None):
        return self.get_model(name=name, training_variant='random_smoothing')

    def get_variant_none(self, model, name=None):
        return self.get_model(name=name, training_variant='none')

    def get_plots(self):
        plots = {
            'vulnerable': (self.model_name + ' SDC', 'accuracy', 'vulnerable', 'hist'),
        }
        return plots

    def vulnerable(self):
        # TODO fix for random smoothing
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
        plt.title(self.get_vulnerable_plot_title())
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
        plt.legend()

    def profile_dropin(self):
        for variant in self.get_variants():
            dropin_model = self.get_variant(None, variant)
            d = Dropin(dropin_model, representative_dataset=self.get_profile_database(dropin_model))
            with open(self.get_profile_path(variant), mode='wb') as f:
                pickle.dump((d.a, d.b), f)
            # dropin 0.0, 972.57635
            # none 0.0 953.709

    def get_profile_path(self, variant):
        return 'dropin/resources/{}_{}_profile.pkl'.format(self.__class__.__name__, variant)

    @abstractmethod
    def get_profile_database(self, dropin_model):
        pass

    @abstractmethod
    def train(self, training_variant='none'):
        pass

    def get_vulnerable_plot_title(self):
        return self.model_name

    def get_model(self, name=None, training_variant='dropin'):
        model = self.get_raw_model(name=name)
        try:
            model.load_weights(self.get_checkpoint_filepath(variant=training_variant))
        except Exception as e:
            logger.error(str(e))
        profile_path = self.get_profile_path(training_variant)
        if os.path.exists(profile_path):
            with open(profile_path, mode='rb') as f:
                a, b = pickle.load(f)
                model.dropin.a, model.dropin.b = a, b
                logger.info("Loaded bounds for dropin as {}".format((a, b)))
        else:
            logger.error("Profile info doesn't exist {}".format(profile_path))
        return model
