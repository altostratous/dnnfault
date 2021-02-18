import logging
import os
import pickle
from abc import abstractmethod

import tfi
from tensorflow.python.keras import Model

logger = logging.getLogger(__file__)


class ExperimentBase:

    epochs = 300
    variants = ('none', )
    default_config = {
        'Artifact': 'convs',
        'Type': 'mutate',
        'Amount': 1,
        'Bit': '23-30',
    }

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.evaluations = []

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
    def get_model(self):
        pass

    def run(self):
        dataset = self.get_dataset()
        for epoch in range(self.epochs):
            logger.info('Started epoch {}'.format(epoch))
            for config_patch in self.get_configs():
                config = self.get_default_config()
                config.update(config_patch)
                logger.info('Injecting fault with config {}'.format(config))
                faulty_model = self.get_faulty_model(config)
                for variant_key in self.get_variants():
                    logger.info('Creating variant {}'.format(variant_key))
                    model = getattr(self, 'get_variant_{}'.format(variant_key))(faulty_model)
                    logger.info('Evaluating ...')
                    self.compile_model(model)
                    evaluation_result_chunk = self.evaluate(model, dataset)
                    logger.info('Saving evaluation ...')
                    self.save_evaluation_chunk(epoch, config, variant_key, evaluation_result_chunk)

    def copy_model(self, faulty_model):
        model = self.get_raw_model()
        model.set_weights(faulty_model.get_weights())
        return model

    @abstractmethod
    def get_configs(self):
        pass

    def get_faulty_model(self, config) -> Model:
        model = self.get_model()
        tfi.inject(fiConf=config, model=model)
        return model

    def get_raw_model(self) -> Model:
        return self.get_model()

    def get_variants(self):
        return self.variants

    def get_variant_none(self, model):
        return self.copy_model(model)

    @abstractmethod
    def evaluate(self, model, dataset):
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
        return 'tmp/' + self.__class__.__name__ + str(epoch) + '.pkl'

    def save_log_object(self, log_object, log_file_name):
        self.evaluations.append(log_object)
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        with open(log_file_name, mode='wb') as f:
            pickle.dump(self.evaluations, f)

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def compile_model(self, model):
        pass
