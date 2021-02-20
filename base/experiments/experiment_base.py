import gc
import logging
import os
import pickle
from abc import abstractmethod
from matplotlib import pyplot as plt

import tfi
from tensorflow.python.keras import Model

logger = logging.getLogger(__file__)


class ExperimentBase:

    epochs = 300
    variants = ('none', )
    actions = (
        'run',
        'plot',
    )
    default_config = {
        'Artifact': 'convs',
        'Type': 'mutate',
        'Amount': 1,
        'Bit': '23-30',
    }
    plots = {}

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.evaluations = []
        self.try_recover_evaluations()

    @staticmethod
    def find_experiments(apps):
        experiments = []
        for app in apps:
            experiments_module = __import__('{}.experiments'.format(app))
            for e in experiments_module.experiments.__dict__.values():
                if isinstance(e, type):
                    if issubclass(e, ExperimentBase):
                        experiments.append(e)
        return experiments

    @abstractmethod
    def get_model(self, name=None):
        pass

    def run(self):
        dataset = self.get_dataset()
        x, y_true = next(iter(dataset))
        counter = 0
        for epoch in range(self.epochs):
            logger.info('Started epoch {}'.format(epoch))
            for config_id, config_patch in enumerate(self.get_configs()):
                if len(self.evaluations) >= counter + len(self.variants):
                    counter += len(self.variants)
                    logger.info('Skipping already done evaluation {} ...'.format(counter))
                    continue
                config = self.get_default_config()
                config.update(config_patch)
                logger.info('Injecting fault with config {}'.format(config))
                faulty_model = self.get_faulty_model(config, name='faulty_{}_{}'.format(
                    epoch,
                    config_id
                ))
                for variant_key in self.get_variants():
                    if len(self.evaluations) > counter:
                        logger.info('Skipping already done variant {} ...'.format(variant_key))
                        counter += 1
                        continue
                    logger.info('Creating variant {}'.format(variant_key))
                    model = getattr(self, 'get_variant_{}'.format(variant_key))(faulty_model,
                                                                                name='variant_{}_{}_{}'.format(
                                                                                    epoch,
                                                                                    config_id,
                                                                                    variant_key
                                                                                ))
                    logger.info('Evaluating ...')
                    self.compile_model(model)
                    evaluation_result_chunk = self.evaluate(model, x, y_true)
                    logger.info('Saving evaluation ...')
                    self.save_evaluation_chunk(epoch, config, variant_key, evaluation_result_chunk)
                    gc.collect()
                    print([o.name for o in gc.get_objects() if isinstance(o, Model)])
                    counter += 1

    def copy_model(self, faulty_model, name=None):
        model = self.get_raw_model(name=name)
        model.set_weights(faulty_model.get_weights())
        return model

    @abstractmethod
    def get_configs(self):
        pass

    def get_faulty_model(self, config, name=None) -> Model:
        model = self.get_model(name=name)
        tfi.inject(fiConf=config, model=model)
        return model

    def get_raw_model(self, name=None) -> Model:
        return self.get_model(name=name)

    def get_variants(self):
        return self.variants

    def get_variant_none(self, model, name=None):
        return self.copy_model(model, name=name)

    @abstractmethod
    def evaluate(self, model, x, y_true):
        pass

    def get_default_config(self):
        return self.default_config.copy()

    def save_evaluation_chunk(self, epoch, config, variant_key, evaluation_result_chunk):
        log_object = self.get_log_file_object(config, epoch, evaluation_result_chunk, variant_key)
        log_file_name = self.get_log_file_name(epoch, config, variant_key)
        self.save_log_object(log_object, log_file_name)

    def get_log_file_object(self, config, epoch, evaluation_result_chunk, variant_key):
        log_object = {
            'epoch': epoch,
            'variant_key': variant_key,
            'config': config,
            'evaluation': evaluation_result_chunk
        }
        return log_object

    def get_log_file_name(self, epoch, config, variant_key):
        return self.get_log_file_name_prefix() + str(epoch) + '.pkl'

    def get_log_file_name_prefix(self):
        return 'tmp/' + self.__class__.__name__

    def save_log_object(self, log_object, log_file_name):
        self.evaluations.append(log_object)
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        with open(log_file_name, mode='wb') as f:
            pickle.dump(self.evaluations, f)

    def load_evaluations(self):
        with open(self.args.data_file_name, mode='rb') as f:
            self.evaluations = pickle.load(f)

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def compile_model(self, model):
        pass

    def plot(self):
        self.load_evaluations()
        plot_key = self.args.plot_key
        title, x_title, y_title = self.plots[plot_key]
        x, y = getattr(self, plot_key)()
        self.draw_plot(x, y, title, x_title, y_title)

    def draw_plot(self, x, y, title, x_title, y_title):
        for y_, variant in zip(y, self.variants):
            y_value, error = y_
            plt.errorbar(x, y_value, error, label=variant, elinewidth=0.5, capsize=5)
        plt.legend()
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()

    def try_recover_evaluations(self):
        log_file_name_prefix = self.get_log_file_name_prefix()
        log_directory = os.path.dirname(log_file_name_prefix)
        os.makedirs(log_directory, exist_ok=True)
        log_files = [os.path.join(log_directory, n) for n in os.listdir(log_directory)
                     if os.path.join(log_directory, n).startswith(log_file_name_prefix)]
        most_recent_log_file_name = self.get_most_recent_log_file_name(log_files)
        if most_recent_log_file_name is None:
            return
        with open(most_recent_log_file_name, mode='rb') as f:
            self.evaluations = pickle.load(f)

    def get_most_recent_log_file_name(self, log_files):
        maximum = None
        maximum_index = None
        for i, log_file in enumerate(log_files):
            epoch = int(log_file[len(self.get_log_file_name_prefix()):].split('.')[0])
            if maximum is None or epoch > maximum:
                maximum = epoch
                maximum_index = i
        if maximum_index is None:
            return None
        return log_files[maximum_index]
