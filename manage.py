import argparse

from base.experiments import ExperimentBase
from settings import APPS
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

parser = argparse.ArgumentParser(description='Command line interface for dnn fault.')

experiments = ExperimentBase.find_experiments(APPS)
experiments_map = {experiment.__name__: experiment for experiment in experiments}
parser.add_argument('experiment', type=str,
                    choices=experiments_map.keys())
parser.add_argument('--action', default='run', type=str)
parser.add_argument('--data_file_name', default='run', type=str)
parser.add_argument('--validation_split', type=float, default=0.5)
parser.add_argument('--subset', type=str, default='training')
parser.add_argument('--plot_key', type=str)

args = parser.parse_args()
experiment = experiments_map[args.experiment](args)
args = parser.parse_args()

getattr(experiment, args.action)()

