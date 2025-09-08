import component as cn
import torch

class LinearRegression(cn.Module):
    def __init__(self, device, *args, **kwargs):
        super(LinearRegression, self).__init__(device)
        self.name = 'LinearRegression'
        self.model = None
        self.input_require_shape = [100]  

    def generate_model(self, input_data_shape, output_dim=1, *args, **kwargs):
        self.output_dim = output_dim if output_dim > 0 else 1
        self.model = LinearRegression_Model(
            input_data_shape, self.output_dim).to(self.device)
        self.create_Loc_reshape_list()

    def fix_Loc_list(self):
        if len(self.Loc_list) > 0:
            self.Loc_list = [self.Loc_list[0]]

class LinearRegression_Model(torch.nn.Module):
    def __init__(self, input_data_shape, output_dim):
        super(LinearRegression_Model, self).__init__()
        input_dim = input_data_shape[0]
        self.predictor = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.predictor(x)