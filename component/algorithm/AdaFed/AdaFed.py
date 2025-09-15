
import component as cn
import torch
import numpy as np
import copy
from component.algorithm.common.utils import Gram_Schmidt
from component.algorithm.common.utils import get_d_adafed
import time


class AdaFed(cn.Algorithm):
    def __init__(self,
                 name='AdaFed',
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
                 power=1,
                 *args,
                 **kwargs):
        if params is not None:
            power = params['pow']

        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' power' + str(power)


        super().__init__(name, data_loader, loader_name, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, epochs, save_name, outFunc, write_log, params)
        self.power = power

    def run(self):
        round = 0
        batch_num = np.mean(self.get_client_attr('training_batch_num'))

        while not self.terminated():
            round += 1
            com_time_start = time.time()

            m_locals, l_locals = self.train()
            com_time_end = time.time()
            cal_time_start = time.time()

            g_locals = []
            old_model = self.module.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                g_locals.append(
                        (old_model - m_locals[idx]) / self.lr)

            g_locals = torch.stack(g_locals)
            g_locals /= (torch.norm(g_locals, dim=1).reshape(-1, 1))
            g_orthogonal = Gram_Schmidt(g_locals, l_locals, self.power)
            d = get_d_adafed(g_orthogonal)
            self.update_module(self.module, self.optimizer, self.lr, d)
            self.client_update()
            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start
