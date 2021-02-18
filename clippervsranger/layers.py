from collections import defaultdict
from tensorflow.keras.layers import Layer
import tensorflow as tf


class ProfileLayer(Layer):
    profile = defaultdict(list)

    def call(self, inputs, **kwargs):
        if hasattr(inputs, 'numpy'):
            self.profile[self.name].append(inputs.numpy())
        return super().call(inputs, **kwargs)


class RangeRestrictionLayer(Layer):

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.bounds = kwargs.pop('bounds')
        super().__init__(trainable, name, dtype, dynamic, **kwargs)


class RangerLayer(RangeRestrictionLayer):

    def call(self, inputs, **kwargs):
        upper = self.bounds[self.name]['upper']
        lower = self.bounds[self.name]['lower']
        return tf.minimum(tf.maximum(super().call(inputs, **kwargs), lower), upper)


class ClipperLayer(RangeRestrictionLayer):

    def call(self, inputs, **kwargs):
        upper = self.bounds[self.name]['upper']
        lower = self.bounds[self.name]['lower']
        result = super().call(inputs, **kwargs)
        mask = tf.logical_or(
            tf.greater(result, upper),
            tf.less(result, lower),
        )
        return tf.where(mask, 0., result)

