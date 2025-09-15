
import component as cn
import os
import torch
import torchvision
import random
from torchvision import transforms as transforms
import numpy as np
import copy


class DataLoader_fashion_dir(cn.DataLoader):

    def __init__(self,
                 pool_size=100,
                 alpha=0.1,
                 batch_size=100,
                 input_require_shape=None,
                 shuffle=True,
                 recreate=False,
                 params=None,
                 device='cpu',
                 *args,
                 **kwargs):

        if params is not None:
            pool_size = params['N']
            alpha = params['Diralpha']
            batch_size = params['B']
            device = params.get('device', 'cpu')

        name = 'Fashion_dir_pool_' + str(pool_size) + 'alpha_' + str(alpha) + '_batchsize_' + str(
            batch_size) + '_sort_split_input_require_shape_' + str(input_require_shape)
        nickname = 'fashion dir B' + \
            str(batch_size) + ' alpha' + str(alpha) + ' N' + str(pool_size)
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
                item_classes_num=None, batch_size=batch_size,
                alpha=alpha, niid=True, partition='dir'
            )
            self.statistic = statistic

            self.data_pool = cn.create_data_pool(
                [torch.tensor(x) for x in X],
                [torch.tensor(y) for y in y],
                pool_size, shuffle, train_prob,
                batch_size, self.target_class_num
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