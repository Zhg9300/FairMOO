
from component.main import initialize, read_params, outFunc
import os


from component.Algorithm import Algorithm
from component.Client import Client
from component.DataLoader import DataLoader
from component.Module import Module
from component.Metric import Metric
from component.seed import setup_seed

from component.metric.Correct import Correct
from component.metric.Precision import Precision
from component.metric.Recall import Recall

from component.model.CNN import CNN
from component.model.MLP import MLP


import component.algorithm
from component.algorithm.FedAvg.FedAvg import FedAvg
from component.algorithm.qFedAvg.qFedAvg import qFedAvg
from component.algorithm.AFL.AFL import AFL
from component.algorithm.FedMGDA_plus.FedMGDA_plus import FedMGDA_plus
from component.algorithm.DRFL.DRFL import DRFL
from component.algorithm.FedMDFG.FedMDFG import FedMDFG
from component.algorithm.FedLF.FedLF import FedLF
from component.algorithm.AdaFed.AdaFed import AdaFed
from component.algorithm.FairMOO.FairMOO import FairMOO

from component.dataloaders.separate_data import separate_data, create_data_pool
from component.dataloaders.DataLoader_cifar10_pat import DataLoader_cifar10_pat
from component.dataloaders.DataLoader_cifar10_dir import DataLoader_cifar10_dir
from component.dataloaders.DataLoader_fashion_pat import DataLoader_fashion_pat
from component.dataloaders.DataLoader_fashion_dir import DataLoader_fashion_dir
from component.dataloaders.DataLoader_synthetic import DataLoader_synthetic

data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)


pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)
