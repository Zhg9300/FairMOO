import component as cn


class Precision(cn.Metric):
    def __init__(self):
        super().__init__(name='precision')

    @staticmethod
    def calc(network_output, target):
        true_positive = ((target * network_output) > .1).int().sum(axis=-1)
        return (true_positive / (network_output.sum(axis=-1) + 1e-13)).sum().item()
