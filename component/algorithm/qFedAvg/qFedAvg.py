
import component as cn
import torch
import time
import numpy as np


class qFedAvg(cn.Algorithm):
    def __init__(self,
                 name='qFedAvg',
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
                 q=0.1,
                 *args,
                 **kwargs):

        if params is not None:
            q = params['q']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' q' + str(q)

        super().__init__(name, data_loader, loader_name, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, epochs, save_name, outFunc, write_log, params)
        self.q = q

        self.lr = self.train_setting['optimizer'].defaults['lr']

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
            old_model_params = self.module.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                grad = (old_model_params -
                        m_locals[idx]) * (1 / self.lr)
                g_locals.append(grad)
            g_locals_mat = torch.stack(g_locals)
            l_locals1 = torch.Tensor(l_locals).float().to(self.device) + 1e-10
            Deltas = l_locals1.reshape(-1, 1)**self.q * g_locals_mat            
            hs = self.q * l_locals1**(self.q - 1) * torch.norm(
                g_locals_mat, dim=1)**2 + 1.0 / self.lr * l_locals1**self.q 
            self.aggregate(old_model_params, Deltas, hs)
            self.client_update()
            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start


    def aggregate(self, old_model_params, Deltas, hs):
        denominator = torch.sum(hs)
        scaled_deltas = Deltas / denominator
        updates = torch.sum(scaled_deltas, dim=0)
        new_params = old_model_params - updates
        self.module.reshape_vec_to_model_params(new_params)
