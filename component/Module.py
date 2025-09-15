
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters


class Module(nn.Module):

    def __init__(self, device, *args, **kwargs):
        super(Module, self).__init__()
        self.name = 'Module'
        self.device = device
        self.input_require_shape = None
        self.ignore_head = False
        self.Loc_reshape_list = None
        self.Loc_list = None
        self.model = None
        self.target_class_num = 0

    def generate_model(self, input_data_shape, target_class_num, *args, **kwargs):
        raise NotImplementedError

    def fix_Loc_list(self):
        pass

    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        self.Loc_list = []
        for i, p in enumerate(self.model.parameters()):
            param_num = p.numel()
            Loc = torch.arange(currentIdx, currentIdx + param_num, 1)
            self.Loc_list.append(Loc)
            self.Loc_reshape_list.append(Loc.reshape(p.data.shape))
            currentIdx += param_num


    @staticmethod
    def change_data_device(data, device):
        new_data = None
        if type(data) == torch.Tensor:
            new_data = data.to(device)
        elif type(data) == tuple:
            new_data = []
            for item in data:
                item = item.to(device)
                new_data.append(item)
            new_data = tuple(new_data)
        elif type(data) == list:
            new_data = []
            for item in data:
                item = item.to(device)
                new_data.append(item)
        return new_data

    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, Module):
            raise TypeError
        res_module = copy.deepcopy(self)
        res_module.model.zero_grad()
        res_model_params = res_module.model.state_dict()
        other_params = other.model.state_dict()
        for layer in res_model_params.keys():
            res_model_params[layer] += other_params[layer]
        return res_module

    def __sub__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, Module):
            raise TypeError
        res_module = copy.deepcopy(self)
        res_module.model.zero_grad()
        res_model_params = res_module.model.state_dict()
        other_params = other.model.state_dict()
        for layer in res_model_params.keys():
            res_model_params[layer] -= other_params[layer]
        return res_module

    def __mul__(self, other):

        res_module = copy.deepcopy(self)
        res_module.model.zero_grad()
        res_model_params = res_module.model.state_dict()
        if not isinstance(other, Module):
            for k in res_model_params.keys():
                res_model_params[k] *= other
        else:
            other_params = other.model.state_dict()
            for k in res_module.model.state_dict().keys():
                res_model_params[k] *= other_params[k]
        return res_module

    def __rmul__(self, other):

        return self * other

    def __pow__(self, power):
        return self._model_norm(power)

    def dot(self, other):

        res = 0.0
        md1 = self.model.state_dict()
        md2 = other.model.state_dict()
        for layer in md1.keys():
            if md1[layer] is None:
                continue

            res += torch.sum(md1[layer] * md2[layer])
        return res

    def dot_vec(self, a, b):
        sums = 0.0
        for layer in self.Loc_list:
            a_layer = a[layer]
            b_layer = b[layer]
            sums += a_layer @ b_layer
        return sums

    def L2_norm_square(self):
        model_params = self.model.state_dict()
        res = 0.0
        for k in model_params.keys():
            if model_params[k] is None:
                continue
            if model_params[k].dtype not in [torch.float, torch.float32, torch.float64]:
                continue
            res += torch.sum(torch.pow(model_params[k], 2))
        return res

    def norm(self, p=2):
        return self**p

    def _model_norm(self, p):

        res = 0.0
        md = self.model.state_dict()
        for layer in md.keys():
            if md[layer] is None:
                continue
            if md[layer].dtype not in [torch.float, torch.float32, torch.float64]:
                continue
            res += torch.sum(torch.pow(md[layer], p))
        return torch.pow(res, 1.0 / p)

    @staticmethod
    def model_sum(module_list):
        res_module = copy.deepcopy(module_list[0])
        res_model_params = res_module.model.state_dict()
        for module in module_list[1:]:
            model_params = module.model.state_dict()
            for k in model_params.keys():
                res_model_params[k] += model_params[k]
        return res_module

    def clone_from(self, module):
        self.model.load_state_dict(module.model.state_dict())

    def span_model_grad_to_vec(self):
        grad_vec = []
        for p in self.model.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.data.clone().flatten())
        return torch.cat(grad_vec) if grad_vec else torch.tensor([])

    def reshape_vec_to_model_grad(self, vec):
        with torch.no_grad():
            pointer = 0
            for p in self.model.parameters():
                if p.requires_grad: 
                    num_params = p.numel()
                    p.grad = vec[pointer:pointer + num_params].view_as(p).clone()
                    pointer += num_params

    def span_model_params_to_vec(self):
        return parameters_to_vector(self.model.parameters()).detach().clone()

    def reshape_vec_to_model_params(self, vec):
        with torch.no_grad():
            vector_to_parameters(vec, self.model.parameters())
