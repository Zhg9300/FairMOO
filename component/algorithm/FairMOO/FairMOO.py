import component as cn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from component.algorithm.common.utils import get_d_FairMOO
import time


class FairMOO(cn.Algorithm):
    def __init__(self,
                 name='FairMOO',
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
                 prefer=6e-3,
                 *args,
                 **kwargs):
        if params is not None:
            prefer = params['prefer']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' prefer' + str(prefer)

        super().__init__(name, data_loader, loader_name, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, epochs, save_name, outFunc, write_log, params)
        self.prefer = prefer

    def run(self):

        batch_num = np.mean(self.get_client_attr('training_batch_num'))
        round = 0
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
            l = torch.tensor(l_locals).to(self.device).float()
            fairness = l.var()
            l = l - l.mean()
            g_fair = 2 / len(l_locals) * (l.unsqueeze(1) * torch.stack(g_locals)).sum(dim=0)
            is_fair = 1 if fairness <= self.prefer else 0
            g_locals = torch.stack(g_locals)
            if is_fair:
                d, sol = get_d_FairMOO(g_locals, self.device, g_fair)
            else:
                d = g_fair
            d /= torch.norm(d)
            self.update_module(self.module, self.optimizer, self.lr, d)
            self.client_update()

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start
