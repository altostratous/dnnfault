import random
import re

from tensorflow.python.keras import Model, Input

from base.utils import insert_layer_nonseq
from dropin.layers import DropinProfiler
import logging
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class Dropin:
    regex = 'conv2d.*|dense.*'

    def __init__(self, model, representative_dataset=None, a=None, b=None) -> None:
        super().__init__()
        self.model = model
        self.representative_dataset = representative_dataset

        if self.representative_dataset:
            def profiler_layer_factory(insert_layer_name):
                return DropinProfiler(name=insert_layer_name)

            profiler = insert_layer_nonseq(model, self.regex, profiler_layer_factory, 'profiler')
            profiler.run_eagerly = True
            train_data_size = tf.data.experimental.cardinality(representative_dataset).numpy()
            for i, data in enumerate(self.representative_dataset):
                x, y = data
                profiler.predict(x)
                logger.info('Done with {}/{} batches.'.format(i, train_data_size))
            self.a, self.b = DropinProfiler.a, DropinProfiler.b
        else:
            self.a, self.b = a, b
        self.perturbation_inputs = []

    def augment_model(self, model: Model):
        x = model.input
        self.model_input = model.input
        for layer in model.layers:
            if re.match(self.regex, layer.name):
                original_output = layer(x)
                perturbation_input = Input(
                    shape=tuple(d for d in original_output.shape if d is not None),
                    name=layer.name + '_perturbation')
                x = original_output + perturbation_input
                self.perturbation_inputs.append(perturbation_input)
            else:
                x = layer(x)
        return Model(inputs=[model.inputs] + self.perturbation_inputs, outputs=x)

    def augment_data(self, data, label=None):
        result = [data]
        weights = [np.prod([d for d in i.shape if d is not None])
                   for i in self.perturbation_inputs]
        weights_sum = sum(weights)
        probabilities = [w / weights_sum / 2 for w in weights] + [1 / 2]

        perturbation_index = np.random.choice(len(self.perturbation_inputs) + 1,
                                              p=probabilities)
        for i, perturbation_input in enumerate(self.perturbation_inputs):
            if i == perturbation_index:
                result.append(self.generate_perturbation(len(data),
                                                         perturbation_input))
            else:
                result.append(np.zeros(
                    (len(data),) + tuple(d for d in perturbation_input.shape
                                         if d is not None)))
        return result

    def augment_zero(self, data, label=None):
        result = [data]
        for i, perturbation_input in enumerate(self.perturbation_inputs):
            result.append(np.zeros(
                (len(data),) + tuple(d for d in perturbation_input.shape
                                     if d is not None)))
        return result

    def get_max_magnitude(self):
        return 2 ** np.ceil(np.log2(np.maximum(np.abs(self.a), np.abs(self.b))))

    def generate_perturbation(self, batch_size, perturbation_input):
        shape = (batch_size,) + tuple(d for d in perturbation_input.shape if d is not None)
        zeroes = np.zeros(shape)
        zeroes = zeroes.T
        if 'conv' in perturbation_input.name:
            channel_to_terminate = random.randrange(zeroes.shape[0])
            zeroes[channel_to_terminate] += (-1) ** random.randint(0, 1) * self.get_max_magnitude()
        elif 'dense' in perturbation_input.name:
            access = zeroes
            while len(access.shape) > 1:
                access = access[random.randrange(len(access))]
            access += (-1) ** random.randint(0, 1) * self.get_max_magnitude()
        zeroes = zeroes.T
        return zeroes
