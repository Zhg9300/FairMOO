
import component as cn
import numpy as np
import time
import torch


class DRFL(cn.Algorithm):
    def __init__(self,
                 name='DRFL',
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
                 dishonest=None,
                 params=None,
                 *args,
                 **kwargs):

        super().__init__(name, data_loader, loader_name, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, epochs, save_name, outFunc, write_log, dishonest, params)

    def run(self):

        batch_num = np.mean(self.get_client_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, l_locals = self.train()
            com_time_end = time.time()
            cal_time_start = time.time()
            l_locals = torch.Tensor(l_locals).float().to(self.device)
            weights = self.online_client_num / self.client_num * l_locals

            self.weight_aggregate(m_locals, weights=weights)
            self.client_update()

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

