import component as cn

class Recall(cn.Metric):
    def __init__(self):
        super().__init__(name='recall')

    @staticmethod
    def calc(network_output, target):
        true_positive = ((target * network_output) > .1).int().sum(axis=-1)
        return (true_positive / (target.sum(axis=-1) + 1e-13)).sum().item()
