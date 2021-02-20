import json
import pickle
import random
from collections import defaultdict

import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.imagenet_utils import CLASS_INDEX_PATH
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow.python.keras.utils import data_utils

from base.experiments import ExperimentBase
from base.utils import insert_layer_nonseq
from clippervsranger.layers import RangerLayer, ClipperLayer


class ClipperVSRangerV3(ExperimentBase):

    plots = {
        'sdc': ('SDC rate', 'bit-flips', 'sdc rate')
    }

    with open('clippervsranger/resources/resnet50bounds.pkl', mode='rb') as f:
        bounds = pickle.load(f)

    variants = (
        'ranger',
        'clipper',
        'none',
        'no_fault',
    )

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

    def get_configs(self):
        target_variables = [
            (layer, variable)
            for layer, variable in enumerate(self.get_model().trainable_variables)
            if 'conv' in variable.name and 'kernel' in variable.name
        ]
        random.seed(0)
        random.shuffle(target_variables)
        for layer, variable in target_variables:
            for conf in [
                {'Amount': 10},
                {'Amount': 1},
                {'Amount': 100},
            ]:
                conf.update({'Artifact': layer})
                yield conf

    def evaluate(self, model, x, y_true):
        y_pred = model.predict(x, batch_size=8)
        return {
            'acc': top_k_categorical_accuracy(y_true, y_pred, k=1),
            'y_true': np.argmax(y_true, axis=1),
            'y_pred': np.argsort(y_pred, axis=1).T[-5:].T
        }

    def get_variant_ranger(self, faulty_model, name=None):
        model = self.copy_model(faulty_model, name=name + '_base_copy')

        def ranger_layer_factory(insert_layer_name):
            return RangerLayer(name=insert_layer_name, bounds=self.bounds)
        model = insert_layer_nonseq(model, '.*relu.*', ranger_layer_factory, 'dummy', model_name=name)
        return model

    def get_variant_no_fault(self, faulty_model, name=None):
        return self.get_model(name=name)

    def get_variant_clipper(self, faulty_model, name=None):
        model = self.copy_model(faulty_model, name=name + '_base_copy')

        def ranger_layer_factory(insert_layer_name):
            return ClipperLayer(name=insert_layer_name, bounds=self.bounds)
        model = insert_layer_nonseq(model, '.*relu.*', ranger_layer_factory, 'dummy', model_name=name)
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
            '../ImageNet-Datasets-Downloader/imagenet/imagenet_images',
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
        pass

    def sdc(self):
        x = [1, 10, 100]
        y = []
        accumulation = {
            1: defaultdict(list),
            10: defaultdict(list),
            100: defaultdict(list),
        }
        for evaluation in self.evaluations:
            accumulation[evaluation['config']['Amount']][evaluation['variant_key']].append(evaluation)

        for variant in self.variants:
            y_ = []
            for amount in accumulation:
                base_correctly_classified = sum(np.sum(np.equal(
                    e['evaluation']['y_pred'].T[-1:][0],
                    e['evaluation']['y_true']
                )) for e in accumulation[amount]['no_fault'])
                changed_to_misclassified = sum(
                    np.sum(np.logical_and(
                        np.equal(accumulation[amount]['no_fault'][i]['evaluation']['y_pred'].T[-1:][0],
                                 accumulation[amount]['no_fault'][i]['evaluation']['y_true']),
                        np.not_equal(e['evaluation']['y_pred'].T[-1:][0],
                                     e['evaluation']['y_true'])
                    ))
                    for i, e in enumerate(accumulation[amount][variant]))
                y_.append(changed_to_misclassified / base_correctly_classified)
            y.append((y_, [0 for _ in y_]))
        return x, y
