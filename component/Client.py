
import component as cn
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
import time
import torch.optim as optim
from torchvision import transforms

from torchvision.datasets import CIFAR100

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863))
])
root = '/path'

class Client:

    def __init__(self,
                 id=None,
                 loader_name=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 *args,
                 **kwargs):
        self.id = id
        self.module = copy.deepcopy(module)
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.loader_name = loader_name,
        self.train_setting = train_setting
        self.metric_list = metric_list
        self.local_training_data = None
        self.local_training_number = 0
        self.local_test_data = None
        self.local_test_number = 0
        self.training_batch_num = 0
        self.test_batch_num = 0
        self.sgd_step = train_setting['sgd_step']

        self.metric_history = {'training_loss': [],
                               'test_loss': [],
                               'local_test_number': 0,
                               }
        for metric in self.metric_list:
            self.metric_history[metric.name] = []
            if metric.name == 'correct':
                self.metric_history['test_accuracy'] = []
        self.criterion = self.train_setting['criterion'].to(self.device)

        self.upload_loss = None
        self.upload_module = None
        self.upload_grad = None
        self.upload_training_acc = None

    def update_data(self,
                    id,
                    local_training_data,
                    local_training_number,
                    local_test_data,
                    local_test_number,
                    ):
        self.id = id
        self.local_training_data = local_training_data
        self.local_training_number = local_training_number
        self.local_test_data = local_test_data
        self.local_test_number = local_test_number
        self.training_batch_num = len(local_training_data)
        self.test_batch_num = len(local_test_data)

    def free_memory(self):
        self.upload_loss = None
        self.upload_module = None
        self.upload_grad = None
        self.upload_training_acc = None

    def get_message(self, msg):

        return_msg = {}

        if msg['command'] == 'cal_loss':
            self.upload_loss = self.cal_loss(self.module)
            return return_msg
        if msg['command'] == 'cal_gradient_loss':
            lr = msg['lr']
            if self.sgd_step:
                self.cal_gradient_loss_sgd(lr)
            else:
                self.cal_gradient_loss(lr)
            return return_msg
        if msg['command'] == 'train':
            epochs = msg['epochs']
            lr = msg['lr']
            if self.sgd_step:
                self.train_SGD(epochs, lr)
            else:
                self.train(epochs, lr)
        if msg['command'] == 'free_memory':
            self.free_memory()
            return return_msg
        if msg['command'] == 'test':
            self.test()
            return return_msg
        if msg['command'] == 'require_loss':

            return_msg['l_local'] = self.upload_loss
            return return_msg
        if msg['command'] == 'require_gradient_loss':

            return_grad = self.upload_grad
            return_loss = self.upload_loss
            return_msg['g_local'] = return_grad
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_client_module':
            return_module = self.module.span_model_params_to_vec()
            return_loss = self.upload_loss
            return_msg['m_local'] = return_module
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_test_result':
            return_msg['metric_history'] = copy.deepcopy(self.metric_history)
            return return_msg
        if msg['command'] == 'require_attribute_value':
            attr = msg['attr']
            return_msg['attr'] = getattr(self, attr)
            return return_msg

    def cal_loss(self, module):
        module.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                out = module.model(batch_x)
                loss = self.criterion(out, batch_y)
                total_loss += loss.item() * batch_y.shape[0]
        return total_loss / self.local_training_number

    def cal_gradient_loss(self, lr=0.1):
        self.module.model.train()
        grad_mat = []
        total_loss = 0
        weights = []
        for step, (batch_x, batch_y) in enumerate(self.local_training_data):
            weights.append(batch_y.shape[0])
            out = self.module.model(batch_x)
            loss = self.criterion(out, batch_y)
            total_loss += loss.item() * batch_y.size(0)
            self.module.model.zero_grad()
            loss.backward()
            grad_vec = self.module.span_model_grad_to_vec()
            grad_mat.append(grad_vec)
        loss = total_loss / self.local_training_number
        weights = torch.Tensor(weights).float().to(self.device)
        weights = weights / torch.sum(weights)

        grad_mat = torch.stack([grad_mat[i] for i in range(len(grad_mat))])

        g = weights @ grad_mat 

        self.upload_grad = g
        self.upload_loss = float(loss)

    def cal_gradient_loss_sgd(self, lr=0.1):
        self.module.model.train()
        weights = []
        sample_idx = int(np.random.choice(len(self.local_training_data), 1))
        (batch_x, batch_y) = self.local_training_data[sample_idx]
        weights.append(batch_y.shape[0])

        out = self.module.model(batch_x)
        loss = self.criterion(out, batch_y)

        self.module.model.zero_grad()
        loss.backward()
        self.upload_grad = self.module.span_model_grad_to_vec()
        self.upload_loss = float(loss)

    def train_SGD(self, epochs, lr):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)
        self.module.model.train()
        for e in range(epochs):
            sample_idx = int(np.random.choice(
                len(self.local_training_data), 1))
            (batch_x, batch_y) = self.local_training_data[sample_idx]
            out = self.module.model(batch_x)
            loss = self.criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.upload_loss = loss.item()

    def train(self, epochs, lr):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)
        self.module.model.train()
        for e in range(epochs):
            total_loss = 0.0
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                if self.loader_name[0] == 'cifar100':
                    batch_x = train_transforms(batch_x)
                self.module.model.train()
                out = self.module.model(batch_x)
                optimizer.zero_grad()
                loss = self.criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_y.size(0)
            total_loss = total_loss / self.local_training_number
            self.upload_loss = total_loss
            self.metric_history['training_loss'].append(self.upload_loss)


    def test(self):
        metric_dict = self.create_metric_dict()
        self.module.model.eval()
        with torch.no_grad():
            self.metric_history['local_test_number'] = self.local_test_number
            for step, (batch_x, batch_y) in enumerate(self.local_test_data):
                out = self.module.model(batch_x)
                loss = self.criterion(out, batch_y)
                metric_dict['test_loss'] += loss.item() * batch_y.size(0)
                for metric in self.metric_list:
                    metric_dict[metric.name] += metric.calc(out, batch_y)


            self.metric_history['test_loss'].append(
                metric_dict['test_loss'] / self.local_test_number)
            for metric in self.metric_list:
                self.metric_history[metric.name].append(
                    metric_dict[metric.name])
                if metric.name == 'correct':
                    self.metric_history['test_accuracy'].append(
                        100 * metric_dict['correct'] / self.local_test_number)

    def model_param_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.requires_grad:
                total_norm += p.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def create_metric_dict(self):
        metric_dict = {'test_loss': 0}
        for metric in self.metric_list:
            metric_dict[metric.name] = 0
        return metric_dict