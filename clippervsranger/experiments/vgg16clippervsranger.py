import pickle

from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.vgg16 import VGG16

from clippervsranger.experiments import ClipperVSRangerBase


class VGG16ClipperVSRanger(ClipperVSRangerBase):

    variants = (
        'ranger',
        'clipper',
        'none',
        'no_fault',
    )

    with open('clippervsranger/resources/vgg16bounds.pkl', mode='rb') as f:
        bounds = pickle.load(f)

    def get_raw_model(self, name=None) -> Model:
        result = VGG16()
        if name is not None:
            result._name = name
        return result

    def get_model(self, name=None):
        result = VGG16(weights='imagenet')
        if name is not None:
            result._name = name
        return result
