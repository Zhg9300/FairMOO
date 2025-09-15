import component as cn
import os
import torch
import torchvision
import numpy as np
from torchvision import transforms as transforms


class DataLoader_cifar10_pat(cn.DataLoader):

    def __init__(self,
                 pool_size=100,
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

        balance_suffix = "balanced" if balance else "unbalanced"
        name = f'CIFAR10_pat_pool_{pool_size}_item_classes_num_{item_classes_num}_batchsize_{batch_size}_{balance_suffix}_sort_split_input_require_shape_{input_require_shape}'
        nickname = f'cifar10 pat {balance_suffix} B{batch_size} NC{item_classes_num} N{pool_size}'

        super().__init__(name, nickname, pool_size, batch_size, input_require_shape)
        self.device = device

        file_path = cn.pool_folder_path + name + '.npy'
        if os.path.exists(file_path) and not recreate:
            data_loader = np.load(file_path, allow_pickle=True).item()
            for attr in data_loader.__dict__:
                setattr(self, attr, getattr(data_loader, attr))
            self.device = device
            self._move_to_device()
            print('Successfully Read the Data Pool.')
        else:
            print('Preparing data.')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])

            trainset = torchvision.datasets.CIFAR10(
                root=cn.data_folder_path, train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(
                root=cn.data_folder_path, train=False, download=True, transform=transform)

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False) # False
            testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)# False

            train_data = next(iter(trainloader))
            test_data = next(iter(testloader))

            dataset_input = torch.cat([train_data[0], test_data[0]]) 
            dataset_label = torch.cat([train_data[1], test_data[1]])
            self.cal_data_shape(dataset_input.shape[1:])  
            self.target_class_num = 10
            self.total_training_number = len(trainset)
            self.total_test_number = len(testset)
            train_prob = len(trainset) / (len(trainset) + len(testset))
            X, y, statistic = cn.separate_data(
                (dataset_input.numpy(), dataset_label.numpy()),
                train_prob, pool_size, self.target_class_num,
                item_classes_num, batch_size, alpha=None,
                niid=True, balance=balance, partition='pat'
            )
            self.statistic = statistic

            self.data_pool = cn.create_data_pool(
                [torch.tensor(x) for x in X],
                [torch.tensor(y) for y in y],
                pool_size,
                shuffle,
                train_prob,
                batch_size,
                self.target_class_num
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