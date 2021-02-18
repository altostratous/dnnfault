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

args = parser.parse_args()

experiments_map[args.experiment](args).run()

