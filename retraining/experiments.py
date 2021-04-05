from base.experiments import ExperimentBase
from retraining.stages import Train, Evaluate, Analyze


class Retraining(ExperimentBase):

    stage_classes = (
        Train,
        Evaluate,
        Analyze
    )
