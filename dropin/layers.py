import numpy as np
from tensorflow.keras.layers import Layer


class DropinProfiler(Layer):
    a = None
    b = None

    def call(self, inputs, **kwargs):
        if hasattr(inputs, 'numpy'):
            matrix = inputs.numpy()
            minimum = np.min(matrix)
            maximum = np.max(matrix)
            if self.__class__.a is None or self.__class__.a > minimum:
                self.__class__.a = minimum
            if self.__class__.b is None or self.__class__.b < maximum:
                self.__class__.b = maximum
        return super().call(inputs, **kwargs)
