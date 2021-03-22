import pickle

from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.vgg16 import VGG16

from clippervsranger.experiments import ClipperVSRangerBase


class VGG16ClipperVSRanger(ClipperVSRangerBase):

    model_name = 'VGG16'

    variants = (
        'ranger',
        'clipper',
        'none',
        'no_fault',
    )

    activation_name_pattern = '.*_conv[\d]|.*fc[\d]'

    bounds_file_name = 'clippervsranger/resources/vgg16bounds.pkl'
    with open(bounds_file_name, mode='rb') as f:
        bounds = pickle.load(f)

    def get_raw_model(self, name=None) -> Model:
        result = VGG16()
        if name is not None:
            result._name = name
        return result

    def get_model(self, name=None, training_variant=None):
        result = VGG16(weights='imagenet')
        if name is not None:
            result._name = name
        return result
