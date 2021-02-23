import pickle

from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.resnet import ResNet50

from clippervsranger.experiments import ClipperVSRangerBase


class ClipperVSRangerV4(ClipperVSRangerBase):

    bounds_file_name = 'clippervsranger/resources/resnet50bounds.pkl'
    pytorch_bounds_file_name = 'clippervsranger/resources/resnet50bounds_pytorch.pkl'
    with open(bounds_file_name, mode='rb') as f:
        bounds = pickle.load(f)

    def get_raw_model(self, name=None) -> Model:
        result = ResNet50()
        if name is not None:
            result._name = name
        return result

    def get_model(self, name=None):
        result = ResNet50(weights='imagenet')
        if name is not None:
            result._name = name
        return result
