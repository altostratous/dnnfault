from base.state_spaces import StateSpaceBase


class RetrainingEvaluationStateSpace(StateSpaceBase):

    def generate_states(self):
        for epoch_id in range(10):
            for k in range(1, 30, 5):
                for model in ('alexnet', 'resnet18'):
                    for variant in ('random_smoothing', 'ber', 'none'):
                        for dataset in ('cifar-10', ):
                            for batch_id in range(40):
                                yield {
                                    'epoch': epoch_id,
                                    'k': k,
                                    'model': model,
                                    'dataset': dataset,
                                    'batch_id': batch_id,
                                    'variant': variant,
                                }