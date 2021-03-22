from dropin.experiments import SimpleDense
import tensorflow as tf
import logging

from dropin.utils import Dropin

logger = logging.getLogger(__name__)


class DropoutDense(SimpleDense):
    def get_model(self, name=None, training_variant='dropin'):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ], name=name)

        try:
            model.load_weights(self.get_checkpoint_filepath(variant=training_variant))
        except Exception as e:
            logger.error(str(e))

        if training_variant == 'dropin':
            a = -80.40828
            b = 84.59567
        else:
            a = -58.21058
            b = 50.994366
        dropin = Dropin(model, a=a, b=b, r=0.1)
        model = dropin.augment_model(model)
        setattr(model, 'dropin', dropin)
        return model
