import json
import pickle

from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.imagenet_utils import CLASS_INDEX_PATH
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import TopKCategoricalAccuracy
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow.python.keras.utils import data_utils

from base.experiments import ExperimentBase
from base.utils import insert_layer_nonseq
from clippervsranger.layers import RangerLayer, ClipperLayer


class ClipperVSRanger(ExperimentBase):

    with open('clippervsranger/resources/resnet50bounds.pkl', mode='rb') as f:
        bounds = pickle.load(f)

    variants = (
        'ranger',
        'clipper',
        'none',
        'no_fault',
    )

    def get_raw_model(self) -> Model:
        return ResNet50()

    def get_model(self):
        return ResNet50(weights='imagenet')

    def get_configs(self):
        for layer, variable in enumerate(self.get_model().trainable_variables):
            if 'conv' in variable.name and 'kernel' in variable.name:
                for conf in [
                    {'Amount': 10},
                    {'Amount': 1},
                    {'Amount': 100},
                ]:
                    conf.update({'Artifact': layer})
                    yield conf

    def evaluate(self, model, dataset):
        return model.evaluate(*next(iter(dataset)), batch_size=64)

    def get_variant_ranger(self, faulty_model):
        model = self.copy_model(faulty_model)

        def ranger_layer_factory(insert_layer_name):
            return RangerLayer(name=insert_layer_name, bounds=self.bounds)
        model = insert_layer_nonseq(model, '.*relu.*', ranger_layer_factory, 'dummy')
        return model

    def get_variant_no_fault(self, faulty_model):
        return self.get_model()

    def get_variant_clipper(self, faulty_model):
        model = self.copy_model(faulty_model)

        def ranger_layer_factory(insert_layer_name):
            return ClipperLayer(name=insert_layer_name, bounds=self.bounds)
        model = insert_layer_nonseq(model, '.*relu.*', ranger_layer_factory, 'dummy')
        return model

    def get_dataset(self):
        fpath = data_utils.get_file(
            'imagenet_class_index.json',
            CLASS_INDEX_PATH,
            cache_subdir='models',
            file_hash='c2c37ea517e94d9795004a39431a14cb')
        with open(fpath) as f:
            class_index = json.load(f)

        class_names = []
        class_titles = []
        for i in range(len(class_index)):
            entry = class_index[str(i)]
            class_names.append(entry[0])
            class_titles.append(entry[1])

        dataset = image_dataset_from_directory(
            '../ImageNet-Datasets-Downloader/imagenet/intel',
            label_mode='categorical',
            class_names=class_names,
            image_size=(224, 224),
            validation_split=0.5,
            subset='training',
            seed=0,
            batch_size=2000
        )
        return dataset

    def compile_model(self, model):
        loss = CategoricalCrossentropy()
        model.compile(
            loss=loss,
            metrics=[TopKCategoricalAccuracy(k=1)],
        )


