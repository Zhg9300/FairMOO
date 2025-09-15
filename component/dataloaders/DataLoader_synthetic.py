import component as cn
import os
import torch
import random
import numpy as np
import copy


class DataLoader_synthetic(cn.DataLoader):

    def __init__(self,
                 pool_size=2,
                 item_classes_num=2,  
                 batch_size=100,
                 balance=True,
                 input_require_shape=None,
                 shuffle=True,
                 recreate=False,
                 params=None,
                 device='cpu',
                 *args,
                 **kwargs):

        if params is not None:
            pool_size = params['N']
            item_classes_num = params['NC']
            batch_size = params['B']
            balance = eval(params['balance'])
            device = params.get('device', 'cpu')

        n_dim = 100
        m_samples = 1000
        noise_std = 0.1

        balance_suffix = "balanced" if balance else "unbalanced"
        name = f'Synthetic__pool_{pool_size}_models_{item_classes_num}_batchsize_{batch_size}_{balance_suffix}_input_require_shape_{input_require_shape}'
        nickname = f'synthetic {balance_suffix} B{batch_size} NC{item_classes_num} N{pool_size}'

        super().__init__(name, nickname, pool_size, batch_size, input_require_shape)
        self.device = device

        file_path = cn.pool_folder_path + name + '.npy'
        if os.path.exists(file_path) and (recreate == False):
            data_loader = np.load(file_path, allow_pickle=True).item()
            for attr in list(data_loader.__dict__.keys()):
                setattr(self, attr, data_loader.__dict__[attr])
            self.device = device
            self._move_to_device()
            print('Successfully Read the Data Pool.')
        else:
            print('Generating synthetic regression data...')

            w1 = torch.randn(n_dim, 1) 
            w2 = -w1  
            X = torch.randn(m_samples, n_dim) 
            y1 = torch.matmul(X, w1).squeeze()
            y2 = torch.matmul(X, w2).squeeze()
            X1 = X.clone()
            X2 = X.clone()

            self.input_data_shape = (n_dim,)
            self.target_class_num = item_classes_num
            self.total_training_number = m_samples * item_classes_num * 0.8
            self.total_test_number = m_samples * item_classes_num * 0.2
            train_prob = 0.8  

            self.data_pool = cn.create_data_pool(
                [X1, X2],
                [y1.unsqueeze(-1), y2.unsqueeze(-1)],
                pool_size, shuffle, train_prob, batch_size, self.target_class_num
            )
            np.save(file_path, self)
            print('Successfully Created the Synthetic Regression Data Pool.')
        self._move_to_device()

    def _move_to_device(self):
        for client in self.data_pool:
            if 'local_training_data' in client:
                client['local_training_data'] = [
                    (x.to(self.device), y.to(self.device))
                    for (x, y) in client['local_training_data']
                ]

            if 'local_test_data' in client:
                client['local_test_data'] = [
                    (x.to(self.device), y.to(self.device))
                    for (x, y) in client['local_test_data']
                ]