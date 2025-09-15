import component as cn
import os
import numpy as np
import random
import torch
import copy
import json
import time
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters

class Algorithm:

    def __init__(self,
                 name='Algorithm',
                 data_loader=None,
                 loader_name=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 metric_list=None,
                 max_comm_round=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 params=None,
                 *args,
                 **kwargs):
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [cn.Client(
                i, loader_name, copy.deepcopy(module), device, train_setting, metric_list) for i in range(client_num)] 
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')
        if online_client_num is None:
            online_client_num = client_num

        choose_client_indices = list(np.random.choice(
            client_num, online_client_num, replace=False))
        self.online_client_list = [client_list[i]
                                   for i in choose_client_indices]
        if client_num > online_client_num:
            print(choose_client_indices)
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay'])
        if max_comm_round is None:
            max_comm_round = 10**5
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.loader_name = loader_name
        self.module = module
        self.train_setting = train_setting
        self.client_num = client_num
        self.client_list = client_list
        self.online_client_num = online_client_num
        self.max_comm_round = max_comm_round
        self.epochs = epochs
        self.save_name = save_name
        self.outFunc = outFunc
        self.current_comm_round = 0
        self.module.model.to(device)
        self.metric_list = metric_list
        self.write_log = write_log
        self.params = params
        self.save_folder = ''

        self.stream_log = ""
        self.comm_log = {'client_metric_history': []}

        self.lr = self.train_setting['optimizer'].defaults['lr']
        self.initial_lr = self.lr

        self.optimizer = train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.module.model.parameters()), lr=self.lr)
        self.optimizer.defaults = train_setting['optimizer'].defaults
        self.conflict_log = []
        self.layer_conflict_log = []
        self.descent_log = []
        self.result_module = None
        self.test_interval = 1
        self.communication_time = 0
        self.computation_time = 0

    def run(self):
        raise RuntimeError(
            'error in Algorithm: This function must be rewritten in the child class.')

    @staticmethod
    def update_learning_rate(optimizer, lr):
        optimizer.defaults['lr'] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate(self):
        self.lr = self.initial_lr * \
            self.train_setting['lr_decay']**self.current_comm_round
        self.update_learning_rate(self.optimizer, self.lr)

    def update_module(self, module, optimizer, lr, g):
        self.update_learning_rate(optimizer, lr)
        optimizer.zero_grad()
        for i, p in enumerate(module.model.parameters()):
            p.grad = g[module.Loc_reshape_list[i]].detach().clone()
        optimizer.step()

    def terminated(self):

        self.adjust_learning_rate()
        if self.current_comm_round == 0:
            print('================ Training starts... ================')
        if self.current_comm_round > 0 and self.current_comm_round % self.test_interval == 0:
            self.test()
            if callable(self.outFunc):
                self.outFunc(self)

        self.order_free_memory()
        if self.current_comm_round >= self.max_comm_round:
            if self.current_comm_round % self.test_interval != 0:
                self.test()
                if callable(self.outFunc):
                    self.outFunc(self)
            return True
        else:
            if self.online_client_num < self.client_num:
                choose_client_indices = list(np.random.choice(
                    self.client_num, self.online_client_num, replace=False))
                self.online_client_list = [self.client_list[i]
                                           for i in choose_client_indices]
            self.current_comm_round += 1
            return False


    def client_update(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        with torch.no_grad():
            for idx, client in enumerate(target_client_list):
                for param1, param2 in zip(self.module.model.parameters(), client.module.model.parameters()):
                    if param2.requires_grad:
                        param2.data.copy_(param1.data)

    def weight_aggregate(self, m_locals, weights=None, update_module=True):
        if weights is None:
            weights = torch.Tensor(self.get_client_attr(
                'local_training_number')).float().to(self.device)
        weights = weights / torch.sum(weights)
        params_mat = torch.stack([m_local for m_local in m_locals])
        aggregate_params = weights @ params_mat

        if update_module:
            self.module.reshape_vec_to_model_params(aggregate_params)

    def order_free_memory(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        for client in target_client_list:
            msg = {'command': 'free_memory'}
            client.get_message(msg)

    def get_loss(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        l_locals = []
        for idx, client in enumerate(target_client_list):

            msg = {'command': 'cal_loss'} 
            client.get_message(msg) 

            msg = {'command': 'require_loss'}
            msg = client.get_message(msg)
            l_locals.append(msg['l_local'])

        l_locals = torch.Tensor(l_locals).float().to(self.device)
        return l_locals

    def evaluate(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        g_locals = []
        l_locals = []
        for idx, client in enumerate(target_client_list):

            msg = {'command': 'cal_gradient_loss', 'lr': self.lr}
            client.get_message(msg)

            msg = {'command': 'require_gradient_loss'}
            msg = client.get_message(msg)
            g_locals.append(msg['g_local'])
            l_locals.append(msg['l_local'])

        g_locals = torch.stack([g_locals[i] for i in range(len(g_locals))])
        l_locals = torch.Tensor(l_locals).float().to(self.device)
        return g_locals, l_locals


    def train(self, target_client_list=None):

        if target_client_list is None:
            target_client_list = self.online_client_list
        m_locals = []
        l_locals = []
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'train', 'epochs': self.epochs, 'lr': self.lr}
            client.get_message(msg)
            msg = {'command': 'require_client_module', 'requires_grad': 'False'}
            msg = client.get_message(msg)
            m_locals.append(msg['m_local'])
            l_locals.append(msg['l_local'])
        return m_locals, l_locals

    def test(self): 

        self.comm_log['client_metric_history'] = []
        for idx, client in enumerate(self.client_list):

            msg = {'command': 'test'}
            client.get_message(msg)

            msg = {'command': 'require_test_result'}
            msg = client.get_message(msg)
            self.comm_log['client_metric_history'].append(
                msg['metric_history']) 

        if self.write_log:
            self.save_log()

    def save_log(self):

        save_dict = {'algorithm name': self.name}
        save_dict['info'] = 'data loader name_' + self.data_loader.name + '_module name_' + self.module.name + '_train setting_' + \
            str(self.train_setting) + '_client num_' + str(self.client_num) + \
            '_max comm round_' + str(self.max_comm_round) + \
            '_epochs_' + str(self.epochs)
        save_dict['communication round'] = self.current_comm_round
        save_dict['test interval'] = self.test_interval
        save_dict['online client ids'] = str(
            [online_client.id for online_client in self.online_client_list])
        save_dict['communication log'] = self.comm_log
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        file_name = self.save_folder + self.save_name + '.json'
        fileObject = open(file_name, 'w')
        fileObject.write(json.dumps(save_dict))
        fileObject.close()
        file_name = self.save_folder + 'log_' + self.save_name + '.log'
        fileObject = open(file_name, 'w')
        fileObject.write(self.stream_log)
        fileObject.close()

    def get_client_attr(self, attr='local_training_number', target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list

        attrs = []
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'require_attribute_value', 'attr': attr}
            msg = client.get_message(msg)
            attrs.append(msg['attr'])
        return attrs

    def cal_vec_angle(self, vec_a, vec_b):
        return float(torch.arccos(vec_a @ vec_b / torch.norm(vec_a) / torch.norm(vec_b))) / float(np.pi) * 180
