# -*- coding: utf-8 -*-
import component as cn
import copy
import numpy as np
import torch
import time
import cvxopt
from cvxopt import matrix

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):

    P = 0.5 * (P + P.T)  # make sure P is symmetric

    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]  
    if G is not None:
        args.extend([matrix(G), matrix(h)])  
        if A is not None:
            args.extend([matrix(A), matrix(b)])  
    cal_time_start = time.time()
    sol = cvxopt.solvers.qp(*args)
    cal_time_end = time.time()
    cal_time = cal_time_end - cal_time_start
    optimal_flag = 1
    if 'optimal' not in sol['status']:
        optimal_flag = 0
    return np.array(sol['x']).reshape((P.shape[1],)), optimal_flag, cal_time


def setup_qp_and_solve(vec, device):
    # use cvxopt to solve QP
    P = vec @ (vec.T)
    P = P.cpu().detach().numpy()

    n = P.shape[0]
    q = np.zeros(n)

    G = - np.eye(n)
    h = np.zeros(n)

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False

    sol, optimal_flag, cal_time = cvxopt_solve_qp(P, q, G, h, A, b)
    sol = torch.from_numpy(sol).float().to(device)
    return sol, optimal_flag, cal_time


def get_FedLF_d_layers(grads, value, add_grads, prefer_vec, Loc_list, device, mode=1, return_fair_grad=False):

    value_norm = torch.norm(value)

    Q = grads

    fair_grad = None
    # new cons
    h_vec = (value @ prefer_vec * value / value_norm - prefer_vec * value_norm) / (value_norm**2)
    h_vec = h_vec.reshape(1, -1)

    if grads.shape[0] == 1 and add_grads is None:
        d = grads.reshape(-1)
        return d, Q, fair_grad

    d = []
    if return_fair_grad:
        fair_grad = []
    cal_time_sum = 0.0
    for i in range(len(Loc_list)):
        Q_layer = Q[:, Loc_list[i]]
        # scale each norm
        Q_layer_norm = torch.norm(Q_layer, dim=1)
        miu = torch.mean(Q_layer_norm)
        Q_layer = Q_layer / Q_layer_norm.reshape(-1, 1) * miu
        # add fair
        fair_grad_layer = h_vec @ Q_layer
        fair_grad_layer = fair_grad_layer / torch.norm(fair_grad_layer) * miu
        Q_layer = torch.cat((Q_layer, fair_grad_layer))
        # add history
        if add_grads is not None:
            add_grad_layer = add_grads[:, Loc_list[i]]
            add_grad_layer = add_grad_layer / torch.norm(add_grad_layer, dim=1).reshape(-1, 1) * miu
            Q_layer = torch.vstack([Q_layer, add_grad_layer])
        # solve
        sol_layer, _, cal_time = setup_qp_and_solve(Q_layer, device)
        d_layer = sol_layer @ Q_layer
        d.append(d_layer)
        cal_time_sum += cal_time
        if return_fair_grad:
            fair_grad.append(fair_grad_layer.reshape(-1))
    d = torch.hstack(d)
    if return_fair_grad:
        fair_grad = torch.hstack(fair_grad)

    return d, Q, fair_grad, cal_time_sum


class FedLF(cn.Algorithm):
    def __init__(self,
                 name='FedLF',
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
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 test_conflicts=False,
                 params=None,
                 *args,
                 **kwargs):

        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay'])
        super().__init__(name, data_loader, loader_name, module, device, train_setting, client_num, client_list, online_client_num, metric_list,
                         max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        self.client_online_round_history = [None] * self.client_num  
        self.client_gradient_history = [None] * self.client_num  

        self.used_history_flag = False 

    def train_a_round(self):

        com_time_start = time.time()
        m_locals, l_locals = self.train()
        # calculate approximate gradients
        g_locals = []
        old_model = self.module.span_model_params_to_vec()
        for idx, client in enumerate(m_locals):
            g_locals.append(
                    (old_model - m_locals[idx]) / self.lr)
        g_locals = torch.stack(g_locals)
        l_locals = torch.Tensor(l_locals).float().to(self.device)

        com_time_end = time.time()

        client_id_list = self.get_clinet_attr('id') 
        add_grads = []
        self.used_history_flag = False
        total_client_num = 0
        for item in self.client_online_round_history:
            if item is not None:
                total_client_num += 1
        if total_client_num > self.online_client_num:  
            for client_id, item in enumerate(self.client_online_round_history):
                if item is not None:
                    if self.current_comm_round - item <= tau:  
                        if client_id not in client_id_list: 
                            add_grads.append(self.client_gradient_history[client_id])
        if len(add_grads) == 0:
            add_grads = None
        else:
            add_grads = torch.vstack(add_grads)
            self.used_history_flag = True

        prefer_vec = torch.Tensor([1.0] * self.online_client_num).float().to(self.device)
        prefer_vec = prefer_vec / torch.norm(prefer_vec)

        d, Q, fair_grad, cal_time = get_FedLF_d_layers(g_locals, l_locals, add_grads, prefer_vec, self.module.Loc_list, self.device)

        weights = torch.Tensor([1 / self.online_client_num] * self.online_client_num).float().to(self.device)
        g_norm = torch.norm(weights @ g_locals)
        d_norm = torch.norm(d)
        d_old = copy.deepcopy(d)
        d = d / d_norm * g_norm

        while torch.max(-(Q @ d)) > 1e-6:
            if torch.norm(d) > d_norm * 2:
                d /= 2
            else:
                d = d_old
                break

        # Update parameters of the model
        final_lr = self.lr
        self.update_module(self.module, self.optimizer, final_lr, d)
        self.client_update()
        self.current_training_num += 1
        last_client_id_list = self.get_clinet_attr('id')
        last_g_locals = copy.deepcopy(g_locals)
        for idx, client_id in enumerate(last_client_id_list):
            self.client_online_round_history[client_id] = self.current_comm_round
            temp = self.client_gradient_history[client_id]
            self.client_gradient_history[client_id] = None
            del temp
            self.client_gradient_history[client_id] = last_g_locals[idx]

        self.communication_time += com_time_end - com_time_start
        self.computation_time += cal_time

    def run(self):
        round = 0
        while not self.terminated():
            self.train_a_round()
            round += 1