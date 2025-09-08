
import component as cn
import os
import torch
import torchvision
import random
from torchvision import transforms as transforms
import numpy as np
import copy


class DataLoader_fashion_pat(cn.DataLoader):

    def __init__(self,
                 pool_size=100,
                 item_classes_num=2, #default 2
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
        balance_suffix = "balanced" if balance else "unbalanced"
        name = f'Fashion_pat_pool_{pool_size}_item_classes_num_{item_classes_num}_batchsize_{batch_size}_{balance_suffix}_sort_split_input_require_shape_{input_require_shape}'
        nickname = f'fashion pat {balance_suffix} B{batch_size} NC{item_classes_num} N{pool_size}'

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
            print('Preparing data...')
            transform = transforms.Compose(
                [transforms.ToTensor()])
            trainset = torchvision.datasets.FashionMNIST(root=cn.data_folder_path, train=True,
                                                         download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=trainset.data.shape[0], shuffle=False, num_workers=1)
            testset = torchvision.datasets.FashionMNIST(root=cn.data_folder_path, train=False,
                                                        download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=testset.data.shape[0], shuffle=False, num_workers=1)

            train_data = next(iter(trainloader))
            test_data = next(iter(testloader))
            dataset_input = torch.cat([train_data[0], test_data[0]])
            dataset_label = torch.cat([train_data[1], test_data[1]])

            self.cal_data_shape(train_data[0].shape[1:])
            dataset_input = dataset_input.reshape(-1, *self.input_data_shape)
            self.target_class_num = 10
            self.total_training_number = len(trainset)
            self.total_test_number = len(testset)

            train_prob = len(trainset) / (len(trainset) + len(testset))
            X, y, statistic = cn.separate_data(
                (dataset_input.numpy(), dataset_label.numpy()),
                train_prob, pool_size, self.target_class_num,
                item_classes_num, batch_size, alpha=None,
                niid=True, balance=balance, partition='pat')
            self.data_pool = cn.create_data_pool(
                [torch.tensor(x) for x in X],
                [torch.tensor(y) for y in y],
                pool_size, shuffle, train_prob, batch_size, self.target_class_num
            )
            np.save(file_path, self)
            print('Successfully Created the Data Pool.')
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