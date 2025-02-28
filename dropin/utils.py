import logging
import random
import re

import numpy as np
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.utils.data_utils import Sequence

from base.utils import insert_layer_nonseq
from dropin.layers import DropinProfiler

logger = logging.getLogger(__name__)


class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, processor=lambda j, k=None: (j, k), augmenter=lambda i: i):
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


class Dropin:

    def __init__(self, model, representative_dataset=None, a=None, b=None, r=0.5, mode='worst',
                 regex='conv2d.*|dense.*', perturb=lambda x, p: x + p, count=1, portion=None) -> None:
        super().__init__()
        self.model = model
        self.representative_dataset = representative_dataset
        self.r = r
        self.mode = mode
        self.regex = regex
        self.perturb = perturb

        self.count = count
        self.portion = portion

        if self.representative_dataset:
            DropinProfiler.a, DropinProfiler.b = None, None

            def profiler_layer_factory(insert_layer_name):
                return DropinProfiler(name=insert_layer_name)

            profiler = insert_layer_nonseq(model, self.regex, profiler_layer_factory, 'profiler', only_last_node=True)
            profiler.run_eagerly = True
            train_data_size = len(representative_dataset)
            for i, data in enumerate(self.representative_dataset):
                x, y = data
                profiler.predict(x)
                logger.info('Done with {}/{} batches.'.format(i, train_data_size))
            self.a, self.b = DropinProfiler.a, DropinProfiler.b
        else:
            assert None not in (a, b)
            self.a, self.b = a, b
        self.perturbation_inputs = []

    def augment_model(self, model: Model):
        x = model.input
        self.model_input = model.input
        for layer in model.layers[:-1]:
            if re.match(self.regex, layer.name):
                original_output = layer(x)
                perturbation_input = Input(
                    shape=tuple(d for d in original_output.shape if d is not None),
                    name=layer.name + '_perturbation')
                x = self.perturb(original_output, perturbation_input)
                self.perturbation_inputs.append(perturbation_input)
            else:
                x = layer(x)
        x = model.layers[-1](x)
        return Model(inputs=[model.inputs] + self.perturbation_inputs, outputs=x, name=model.name)

    def augment_data(self, data, label=None):
        result = [data]
        if self.mode == 'zero':
            zeros = np.ones
        else:
            zeros = np.zeros
        weights = [np.prod([d for d in i.shape if d is not None])
                   for i in self.perturbation_inputs]
        weights_sum = sum(weights)
        probabilities = [w / weights_sum * self.r for w in weights] + [1 - self.r]

        perturbation_index = np.random.choice(len(self.perturbation_inputs) + 1,
                                              p=probabilities)
        for i, perturbation_input in enumerate(self.perturbation_inputs):
            if i == perturbation_index:
                result.append(self.generate_perturbation(len(data),
                                                         perturbation_input))
            else:
                result.append(zeros(
                    (len(data),) + tuple(d for d in perturbation_input.shape
                                         if d is not None)))
        return result

    def augment_zero(self, data, label=None):
        result = [data]
        if self.mode == 'zero':
            zeros = np.ones
        else:
            zeros = np.zeros
        for i, perturbation_input in enumerate(self.perturbation_inputs):
            result.append(zeros(
                (len(data),) + tuple(d for d in perturbation_input.shape
                                     if d is not None)))
        return result

    def get_max_magnitude(self):
        return 2 ** self.get_maximum_exponent()

    def generate_perturbation(self, batch_size, perturbation_input):
        shape = (batch_size,) + tuple(d for d in perturbation_input.shape if d is not None)
        if self.mode == 'zero':
            zeros = np.ones
        else:
            zeros = np.zeros
        zeroes = zeros(shape)
        zeroes = zeroes.T
        for _ in range(self.count):
            if (
                'conv' in perturbation_input.name or
                'batch_normalization' in perturbation_input.name
            ):
                dim = zeroes.shape[0]
                if self.portion:
                    channel_to_terminate = random.choices(range(dim), k=int(self.portion * dim))
                else:
                    channel_to_terminate = random.randrange(dim)
                zeroes[channel_to_terminate] = self.perturb(
                    zeroes[channel_to_terminate],
                    (-1) ** random.randint(0, 1) * self.get_magnitude()
                )
            elif 'dense' in perturbation_input.name:
                _access = None
                index = None
                if self.portion is not None:
                    raise ValueError
                access = zeroes
                while len(access.shape) > 1:
                    _access, index = access, random.randrange(len(access))
                    access = access[index]
                _access[index] = self.perturb(access, (-1) ** random.randint(0, 1) * self.get_magnitude())
        zeroes = zeroes.T
        return zeroes

    def get_magnitude(self):
        if self.mode == 'worst':
            return self.get_max_magnitude()
        elif self.mode == 'random':
            return 2 ** random.choice(range(int(self.get_maximum_exponent())))
        elif self.mode == 'zero':
            return 0
        else:
            raise ValueError

    def get_maximum_exponent(self):
        return np.ceil(np.log2(np.maximum(np.abs(self.a), np.abs(self.b))))
