import abc
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class ParameterBase(abc.ABC):

    version = 1
    storage = 'tmp/parameters/'

    def __init__(self, key) -> None:
        super().__init__()
        self.key = key

    def try_evaluate(self):
        last_result = self.load()
        if last_result is not None:
            logger.info('Skipping {}'.format(self.key))
            return
        self.evaluate()

    @abc.abstractmethod
    def evaluate(self):
        pass

    def load(self):
        file_path = self.get_file_path()
        if not os.path.exists(file_path):
            return None
        try:
            return pickle.load(open(file_path, mode='rb'))
        except pickle.UnpicklingError:
            return None

    def get_file_path(self):
        values = [self.__class__.__name__]
        for key, value in sorted(self.key.items()):
            values.append(value)
        return os.path.join(self.storage, str(self.version), ':'.join(values) + '.pkl')
