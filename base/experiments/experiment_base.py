import gc
import logging
import os
import pickle
from abc import abstractmethod

from google.auth.transport import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from matplotlib import pyplot as plt
import numpy as np

import tfi
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K

logger = logging.getLogger(__file__)


class ExperimentBase:

    sheet_id = '15a4LgoXUnpamQ4ufUzZp-75V4vb7AXklGKRarxKoG0w'
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

    table_scheme = {
        'epoch': ('epoch', ),
        'amount': ('config', 'Amount'),
        'layer': ('config', 'Artifact'),
        'variant': ('variant_key', ),
    }

    def get_plots(self):
        return self.plots

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
                if len(self.evaluations) >= counter + len(self.get_variants()):
                    counter += len(self.get_variants())
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
                    evaluation_result_chunk = self.evaluate(model, x, y_true, config)
                    logger.info('Saving evaluation ...')
                    self.save_evaluation_chunk(epoch, config, config_id, variant_key, evaluation_result_chunk)
                    gc.collect()
                    logger.debug(
                        'Resident models {}'.format(', '.join(
                            [o.name for o in gc.get_objects() if isinstance(o, Model)]
                        ))
                    )
                    counter += 1
                K.clear_session()

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
        if self.args.shard:
            return [v for v in self.variants if v == self.args.shard]
        else:
            return self.variants

    def get_variant_none(self, model, name=None):
        return self.copy_model(model, name=name)

    @abstractmethod
    def evaluate(self, model, x, y_true, config):
        pass

    def get_default_config(self):
        return self.default_config.copy()

    def save_evaluation_chunk(self, epoch, config, config_id, variant_key, evaluation_result_chunk):
        log_object = self.get_log_file_object(config, epoch, evaluation_result_chunk, variant_key)
        log_file_name = self.get_log_file_name(epoch, config_id, variant_key)
        self.save_log_object(log_object, log_file_name)

    def get_log_file_object(self, config, epoch, evaluation_result_chunk, variant_key):
        log_object = {
            'epoch': epoch,
            'variant_key': variant_key,
            'config': config,
            'evaluation': evaluation_result_chunk
        }
        return log_object

    def get_log_file_name(self, epoch, config_id, variant_key):
        return self.get_log_file_name_prefix() + str(epoch * 10 + config_id % 2) + '.pkl'

    def get_log_file_name_prefix(self, shard=None):
        if shard is None:
            shard = self.args.shard
        if shard:
            shard_part = '_' + shard + '_'
        else:
            shard_part = ''
        return 'tmp/' + self.args.tag + shard_part + self.__class__.__name__

    def save_log_object(self, log_object, log_file_name):
        self.evaluations.append(log_object)
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        with open(log_file_name, mode='wb') as f:
            pickle.dump(self.evaluations, f)

    def load_evaluations(self):
        if self.args.data_file_name:
            with open(self.args.data_file_name, mode='rb') as f:
                self.evaluations = pickle.load(f)
        else:
            self.try_recover_evaluations()

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def compile_model(self, model):
        pass

    def plot(self):
        self.load_evaluations()
        plot_key = self.args.plot_key
        title, x_title, y_title, pyplot_func = self.get_plots()[plot_key]
        x, y = getattr(self, plot_key)()
        self.draw_plot(x, y, title, x_title, y_title, pyplot_func)

    def draw_plot(self, x, sub_plots, title, x_title, y_title, pyplot_func):
        fig, axe = plt.subplots(len(sub_plots), 1, figsize=(10, 8), sharex='row', sharey='row')
        for sub_plot, y in enumerate(sub_plots.items()):
            condition, y = y
            ax = plt.subplot(len(sub_plots), 1, sub_plot + 1)
            if sub_plot != len(sub_plots) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            for y_, variant in zip(y, self.get_variants()):
                y_value, error = y_
                getattr(self, pyplot_func)(x, y_value, yerr=error, label=variant,
                                           label_index=self.get_variants().index(variant))
            plt.legend()
            plt.title(title + ' ({})'.format(condition) if condition else '')
            plt.ylabel(y_title)
            if isinstance(x[0], str):
                ax.set_xticks(list(range(len(x))))
            else:
                ax.set_xticks(x)
            ax.set_xticklabels(x)
        plt.xticks(rotation='vertical')
        plt.xlabel(x_title)
        plt.tight_layout()

        if self.args.pyplot_out:
            plt.savefig(self.args.pyplot_out)
        else:
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
        logger.info('Recovering evaluations from {}'.format(most_recent_log_file_name))
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

    def export_to_google_sheet(self):
        """Shows basic usage of the Sheets API.
        Prints values from a sample spreadsheet.
        """
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        SAMPLE_RANGE_NAME = 'A1:AA1000'
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        data = self.get_tabular_data()
        response_date = service.spreadsheets().values().update(
            spreadsheetId=self.sheet_id,
            valueInputOption='RAW',
            range=self.get_sheet_name() + '!' + 'A1:AA{}'.format(len(data)),
            body=dict(
                majorDimension='ROWS',
                values=data)
        ).execute()
        logger.info('Sheet successfully Updated')
        logger.info(response_date)

    def get_tabular_data(self):
        self.try_recover_evaluations()
        tabular = [list(self.get_tabular_headers())]
        first_base_evaluation = self.get_first_base_evaluation()
        for evaluation in self.evaluations:
            row = []
            for cell in self.get_table_scheme().items():
                row.append(self.get_tabular_value(cell, evaluation, first_base_evaluation))
            tabular.append(row)
        return tabular

    def get_sheet_name(self):
        return self.__class__.__name__

    def get_tabular_headers(self):
        return self.get_table_scheme().keys()

    def get_table_scheme(self):
        return self.table_scheme

    def get_tabular_value(self, cell, evaluation, first_base_evaluation):
        cell_title, scheme = cell
        if isinstance(scheme, tuple):
            value = evaluation
            while scheme:
                value = value[scheme[0]]
                scheme = scheme[1:]
            return value
        if isinstance(scheme, str):
            return getattr(self, scheme)(cell_title, evaluation, first_base_evaluation)
        raise ValueError

    @abstractmethod
    def get_first_base_evaluation(self):
        pass

    def errorbar(self, x, y, yerr, label, **kwargs):
        plt.errorbar(x, y, yerr=yerr, label=label, elinewidth=0.5, capsize=5)

    def bar(self, x, y, yerr, label, **kwargs):
        variants_count = len(self.get_variants())
        step = 1. / (variants_count + 1)
        plt.bar(np.array(list(range(len(x)))) + (kwargs['label_index'] * step), y, yerr=yerr, label=label, capsize=5, width=step)

    def train(self):
        model = self.get_model()
        self.compile_model(model)
        dataset = self.get_dataset()
        model.fit(dataset)
