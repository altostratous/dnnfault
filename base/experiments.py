import logging
logger = logging.getLogger(__file__)


class ExperimentBase:

    stage_classes = ()

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.stages = {}
        for stage_class in self.stage_classes:
            stage = stage_class(self.args)
            self.stages[stage.name] = stage

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

    def run(self):
        merged_results = []
        for stage_name, stage in self.stages:
            if stage_name == self.args.action:
                self.stages[self.args.action].run(merged_results)
            else:
                merged_results.append(stage.merge())
