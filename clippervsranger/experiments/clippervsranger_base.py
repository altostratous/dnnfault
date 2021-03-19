import json
import re

import tensorflow as tf

import pickle
import random
from abc import ABCMeta
from collections import defaultdict

import numpy as np
from tensorflow.python.keras.applications.imagenet_utils import CLASS_INDEX_PATH
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import top_k_categorical_accuracy, TopKCategoricalAccuracy
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow.python.keras.utils import data_utils

from base.experiments import ExperimentBase
from base.utils import insert_layer_nonseq
from clippervsranger.layers import RangerLayer, ClipperLayer, ProfileLayer
from matplotlib import pyplot as plt


class ClipperVSRangerBase(ExperimentBase, metaclass=ABCMeta):

    z = 1.96  # 95%

    variants = (
        'ranger',
        'clipper',
        'none',
        'no_fault',
    )

    bounds_file_name = None
    pytorch_bounds_file_name = None
    bounds = None

    activation_name_pattern = '.*relu.*|conv[\d]_block[\d]_out'

    model_name = None

    def get_plots(self):
        plots = {
            'sdc': (self.model_name + ' SDC', 'bit-flips', 'sdc', 'errorbar'),
            'class_sdc': (self.model_name + ' Class-wise SDC', 'class index', 'sdc', 'bar'),
            'sorted_class_sdc': (self.model_name + ' Class-wise SDC', 'class index', ('sdc', 'accuracy'), 'bar'),
            'layer_sdc': (self.model_name + ' Layer-wise SDC', 'layer index', 'sdc', 'errorbar'),
            'channel_redundancy': (self.model_name + ' Channel Redundancy (fired channels)', 'layers', 'avg fired channels (> (min + max) / 2)', 'bar'),
            'accuracy_sdc': (self.model_name + ' Channel Redundancy', 'class index', 'redundancy', 'bar'),
        }
        return plots

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

    def evaluate(self, model, x, y_true, config):
        y_pred = model.predict(x, batch_size=64)
        return {
            'acc': top_k_categorical_accuracy(y_true, y_pred, k=1),
            'y_true': np.argmax(y_true, axis=1),
            'y_pred': np.argsort(y_pred, axis=1).T[-5:].T
        }

    def get_variant_ranger(self, faulty_model, name=None):
        model = self.copy_model(faulty_model, name=name + '_base_copy')

        def ranger_layer_factory(insert_layer_name):
            return RangerLayer(name=insert_layer_name, bounds=self.bounds)
        model = insert_layer_nonseq(model, self.activation_name_pattern, ranger_layer_factory, 'dummy', model_name=name)
        return model

    def get_variant_no_fault(self, faulty_model, name=None):
        return self.get_model(name=name)

    def get_variant_clipper(self, faulty_model, name=None):
        model = self.copy_model(faulty_model, name=name + '_base_copy')

        def ranger_layer_factory(insert_layer_name):
            return ClipperLayer(name=insert_layer_name, bounds=self.bounds)
        model = insert_layer_nonseq(model, self.activation_name_pattern, ranger_layer_factory, 'dummy', model_name=name)
        return model

    def get_variant_profiler(self, faulty_model, name=None):
        model = self.copy_model(faulty_model, name=(name or '') + '_base_copy')

        def ranger_layer_factory(insert_layer_name):
            return ProfileLayer(name=insert_layer_name)
        model = insert_layer_nonseq(model, self.activation_name_pattern, ranger_layer_factory, 'dummy', model_name=name)
        return model

    def compile_model(self, model):
        loss = CategoricalCrossentropy()
        model.compile(
            loss=loss,
            metrics=[TopKCategoricalAccuracy(k=1)],
        )

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
                target_evaluations = accumulation[amount][variant]
                changed_to_misclassified = sum(
                    np.sum(np.logical_and(
                        np.equal(accumulation[amount]['no_fault'][i]['evaluation']['y_pred'].T[-1:][0],
                                 accumulation[amount]['no_fault'][i]['evaluation']['y_true']),
                        np.not_equal(e['evaluation']['y_pred'].T[-1:][0],
                                     e['evaluation']['y_true'])
                    ))
                    for i, e in enumerate(target_evaluations))
                p = changed_to_misclassified / base_correctly_classified
                n = len(target_evaluations)
                y_.append((p, self.z * np.sqrt(p * (1 - p) / n)))
            y.append(list(zip(*y_)))
        return x, {'': y}

    def layer_sdc(self):

        subplots = {}
        m = self.get_model()
        variable_names = {}
        for i, t in enumerate(m.trainable_variables):
            if 'conv' in t.name and 'kernel' in t.name:
                variable_names[i] = re.match(r'.*?_?(\d+)_conv', t.name).groups()[0] + '-' + str(int(tf.size(t)))
        for amount in (1, 10, 100):
            y = []
            accumulation = defaultdict(lambda: defaultdict(list))
            for evaluation in self.evaluations:
                if evaluation['config']['Amount'] != amount:
                    continue
                accumulation[evaluation['config']['Artifact']][evaluation['variant_key']].append(evaluation)
            layer_keys = sorted(accumulation.keys())
            for variant in self.variants:
                y_ = []
                for artifact in layer_keys:
                    base_correctly_classified = sum(np.sum(np.equal(
                        e['evaluation']['y_pred'].T[-1:][0],
                        e['evaluation']['y_true']
                    )) for e in accumulation[artifact]['no_fault'])
                    target_evaluations = accumulation[artifact][variant]
                    changed_to_misclassified = sum(
                        np.sum(np.logical_and(
                            np.equal(accumulation[artifact]['no_fault'][i]['evaluation']['y_pred'].T[-1:][0],
                                     accumulation[artifact]['no_fault'][i]['evaluation']['y_true']),
                            np.not_equal(e['evaluation']['y_pred'].T[-1:][0],
                                         e['evaluation']['y_true'])
                        ))
                        for i, e in enumerate(target_evaluations))
                    p = changed_to_misclassified / base_correctly_classified
                    n = len(target_evaluations)
                    y_.append((p, self.z * np.sqrt(p * (1 - p) / n)))
                y.append(list(zip(*y_)))
            subplots['Amount={}'.format(amount)] = y
        return ['-'.join([str(k), variable_names[k]]) for k in layer_keys], subplots

    def class_sdc(self):
        subplots = {}
        for amount in (1, 10, 100):
            y = []
            classes = set()
            accumulation = defaultdict(lambda: defaultdict(list))
            for evaluation in self.evaluations:
                if evaluation['config']['Amount'] != amount:
                    continue
                accumulation[(evaluation['epoch'], evaluation['config']['Artifact'])][evaluation['variant_key']].append(evaluation)
            class_sdc = defaultdict(lambda: defaultdict(int))
            class_base_corrects = defaultdict(lambda: defaultdict(int))
            for experiment in accumulation.values():
                base_line = experiment['no_fault'][0]['evaluation']['y_pred']
                oracle = experiment['no_fault'][0]['evaluation']['y_true']
                for variant, evaluations in experiment.items():
                    for evaluation in evaluations:
                        for i, sample_pred in enumerate(base_line):
                            sample_class = oracle[i]
                            classes.add(sample_class)
                            if base_line[i][-1] != sample_class:
                                continue
                            class_base_corrects[variant][sample_class] += 1
                            if evaluation['evaluation']['y_pred'][i][-1] != sample_class:
                                class_sdc[variant][sample_class] += 1

            classes = sorted(list(classes))
            for variant in self.get_variants():
                y_ = []
                for c in classes:
                    p = class_sdc[variant][c] / class_base_corrects[variant][c]
                    n = class_base_corrects[variant][c]
                    y_.append((p, self.z * np.sqrt(p * (1 - p) / n)))
                y.append(list(zip(*y_)))
            _, titles = self.get_classes_info()
            subplots['Amount={}'.format(amount)] = y
        return [titles[c] for c in classes], subplots

    def sorted_class_sdc(self):
        good_classes, population, subplots, titles, total_population = self.get_sorted_class_sdc_info()
        result = [titles[c] for c in good_classes], {
            **{
                key: [
                    [[variant[0][c] for c in good_classes],
                     [variant[1][c] for c in good_classes]]
                    for variant in y] for key, y in subplots.items() if key == 'Amount=10'
            },
            'Accuracy': [[
                [population[c] / total_population[c] for c in good_classes],
                [0 for _ in good_classes],
            ]]
        }
        return result

    def accuracy_sdc(self):
        good_classes, population, subplots, titles, total_population = self.get_sorted_class_sdc_info(cut_off=500)
        if self.args.filter:
            filters = dict(f.split(':') for f in self.args.filter.split(','))
            x = [subplots['Amount=10'][self.get_variants().index(filters['variant'])][0][c] for c in good_classes]
        else:
            x = [subplots['Amount=10'][self.get_variants().index('clipper')][0][c] for c in good_classes]
        y = [population[c] / total_population[c] for c in good_classes]
        plt.scatter(x, y)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r")
        plt.xlabel('SDC (Amount=10)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy-SDC correlation ({}) for 168 classes'.format(np.corrcoef(x, y)[0][1]))
        plt.show()

    def get_sorted_class_sdc_info(self, cut_off=10):
        subplots = {}
        good_classes = set()
        population = {}
        total_population = {}
        for amount in (1, 10, 100):
            y = []
            classes = set()
            accumulation = defaultdict(lambda: defaultdict(list))
            for evaluation in self.evaluations:
                if evaluation['config']['Amount'] != amount:
                    continue
                accumulation[(evaluation['epoch'], evaluation['config']['Artifact'])][evaluation['variant_key']].append(
                    evaluation)
            class_sdc = defaultdict(lambda: defaultdict(int))
            class_base_corrects = defaultdict(lambda: defaultdict(int))
            class_base_population = defaultdict(lambda: defaultdict(int))
            for experiment in accumulation.values():
                base_line = experiment['no_fault'][0]['evaluation']['y_pred']
                oracle = experiment['no_fault'][0]['evaluation']['y_true']
                for variant, evaluations in experiment.items():
                    for evaluation in evaluations:
                        for i, sample_pred in enumerate(base_line):
                            sample_class = oracle[i]
                            classes.add(sample_class)
                            class_base_population[variant][sample_class] += 1
                            if base_line[i][-1] != sample_class:
                                continue
                            class_base_corrects[variant][sample_class] += 1
                            if evaluation['evaluation']['y_pred'][i][-1] != sample_class:
                                class_sdc[variant][sample_class] += 1

            classes = sorted(list(classes))
            for variant in self.get_variants():
                y_ = []
                for c in classes:
                    n = class_base_corrects[variant][c]
                    total = class_base_population[variant][c]
                    population[c] = n
                    total_population[c] = total
                    wrongs = class_sdc[variant][c]
                    if not n:
                        y_.append((0, 0))
                        continue
                    p = wrongs / class_base_corrects[variant][c]
                    y_.append((p, self.z * np.sqrt(p * (1 - p) / n)))
                    if wrongs:
                        if amount == 1:
                            good_classes.add(c)
                y.append(list(zip(*y_)))
            _, titles = self.get_classes_info()
            subplots['Amount={}'.format(amount)] = y
        good_classes = list(sorted(good_classes))
        good_classes = good_classes[:cut_off] + good_classes[-cut_off:]
        _, good_classes = list(
            zip(*sorted(zip([subplots['Amount=10'][1][0][c] for c in good_classes], good_classes), reverse=True)))
        return good_classes, population, subplots, titles, total_population

    def get_dataset(self):
        class_names, class_titles = self.get_classes_info()
        dataset = image_dataset_from_directory(
            self.get_dataset_path(),
            label_mode='categorical',
            class_names=class_names,
            image_size=(224, 224),
            validation_split=self.args.validation_split,
            subset=self.args.subset,
            seed=0,
            batch_size=10000
        )
        return dataset

    def get_dataset_path(self):
        return self.args.dataset_path or '../ImageNet-Datasets-Downloader/imagenet/imagenet_images'

    def get_classes_info(self):
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
        return class_names, class_titles

    def profile(self):
        model = self.get_variant_profiler(self.get_model())
        self.compile_model(model)
        model.run_eagerly = True
        for x, y in self.get_dataset():
            model.evaluate(x, y)
        bounds = {n: {'upper': max(map(np.max, p)), 'lower': min(map(np.min, p))} for n, p in
                  ProfileLayer.profile.items()}
        with open(self.bounds_file_name, mode='wb') as f:
            pickle.dump(bounds, f)
        print(bounds)

    def tfvspytorch(self):
        x, y = [], []
        pytorch_bounds = pickle.load(open(self.pytorch_bounds_file_name, mode='rb'))
        bounds = pickle.load(open(self.bounds_file_name, mode='rb'))
        for k in sorted(bounds.keys(), key=lambda j: len(j)):
            x.append(pytorch_bounds[k]['upper'])
            y.append(bounds[k]['upper'])
        plt.scatter(x, y)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r")
        plt.xlabel('pytorch bounds')
        plt.ylabel('TensorFlow bounds')
        plt.title('Bounds Correlation across libraries ({})'.format(np.corrcoef(x, y)[0][1]))
        plt.show()

    def get_class_evaluation_sdc(self, class_key, evaluation, base_evaluation):
        class_index = int(class_key.split(':')[0])
        class_sample_indices = [i for i, p in enumerate(base_evaluation['evaluation']['y_true'])
                                if p == class_index and base_evaluation['evaluation']['y_pred'][i][-1] == class_index]
        faulty_count = 0.
        for index in class_sample_indices:
            if evaluation['evaluation']['y_pred'][index][-1] != class_index:
                faulty_count += 1
        return faulty_count / len(class_sample_indices)

    def get_table_scheme(self):
        scheme = super().get_table_scheme().copy()
        first_base_evaluation = self.get_first_base_evaluation()
        classes = set(first_base_evaluation['evaluation']['y_true'])
        class_keys = []
        _, class_titles = self.get_classes_info()
        for i, class_title in enumerate(class_titles):
            if i in classes:
                total_count = len([p for p in first_base_evaluation['evaluation']['y_true']
                                   if p == i])
                correct_count = len([p for j, p in enumerate(first_base_evaluation['evaluation']['y_true'])
                                     if p == first_base_evaluation['evaluation']['y_pred'][j][-1] and p == i])
                class_keys.append('{}: {}, {}/{} correct'.format(i, class_title, correct_count, total_count))
        for class_key in class_keys:
            scheme[class_key] = 'get_class_evaluation_sdc'
        return scheme

    def get_first_base_evaluation(self):
        first_base_evaluation = [e for e in self.evaluations[:len(self.get_variants())]
                                 if e['variant_key'] == 'no_fault'][0]
        return first_base_evaluation

    def channel_redundancy(self):
        names, titles = self.get_classes_info()
        self.variants = ('diamondback', 'wool')
        classes = [titles.index('diamondback'), titles.index('wool')]
        class_names = [names[c] for c in classes]
        dataset = image_dataset_from_directory(
            self.get_dataset_path(),
            label_mode='categorical',
            class_names=class_names,
            image_size=(224, 224),
            validation_split = self.args.validation_split,
            subset=self.args.subset,
            seed=0,
            batch_size=2000
        )
        classes = [0, 1]
        subplots = []

        m = self.get_model()
        for c in classes:
            data = []
            subplot = [[], []]
            for x, y in dataset:
                for x_, y_ in zip(x, y):
                    if c == np.argmax(y_):
                        data.append(x_)
            profiler_model = self.get_variant_profiler(m)
            self.compile_model(profiler_model)
            profiler_model.run_eagerly = True
            ProfileLayer.activated_channels.clear()
            profiler_model.predict(np.array(data), batch_size=len(data))

            for _, r in ProfileLayer.activated_channels.items():
                subplot[0].append(sum(r) / len(r))
                subplot[1].append(0)
            subplots.append(subplot)
        return [l for l, _ in enumerate(ProfileLayer.activated_channels.items())], {'': subplots}

    def get_legend(self, variant, sub_plot, sub_plots_length):
        if sub_plot == 0:
            return super().get_legend(variant, sub_plot, sub_plots_length)
        else:
            return None

