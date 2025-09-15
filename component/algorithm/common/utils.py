import torch
import numpy as np
import cvxopt
from cvxopt import matrix
from scipy.optimize import minimize
import time
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):

    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    optimal_flag = 1
    if 'optimal' not in sol['status']:
        optimal_flag = 0
    return np.array(sol['x']).reshape((P.shape[1],)), optimal_flag


def cvxopt_solve_lp(c, G=None, h=None, A=None, b=None):
    c = c.astype(np.double)
    args = [matrix(c)] 
    if G is not None:
        G = G.astype(np.double)
        h = h.astype(np.double)
        args.extend([matrix(G), matrix(h)])

        if A is not None:
            A = A.astype(np.double)
            b = b.astype(np.double)
            args.extend([matrix(A), matrix(b)])

    sol = cvxopt.solvers.lp(*args)
    optimal_flag = 1 if 'optimal' in sol['status'] else 0

    if sol['x'] is not None:
        x = np.array(sol['x']).reshape((c.shape[0],))
    else:
        x = np.full(c.shape[0], np.nan)  

    return x, optimal_flag

def setup_qp_and_solve(vec):
    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.zeros(n)

    G = - np.eye(n)
    h = np.zeros(n)

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False

    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag


def setup_qp_and_solve_for_mgdaplus(vec, epsilon, lambda0):

    P = np.dot(vec, vec.T)

    n = P.shape[0]
    q = np.zeros(n)

    G = np.vstack([-np.eye(n), np.eye(n)])
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    h = np.hstack([lb, ub])

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag


def quadprog(P, q, G, h, A, b):
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])

def linprog(c, G, h, A, b):
    c = cvxopt.matrix(c.T.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.lp(c, G, h, A.T, b)
    return np.array(sol['x'])

def setup_qp_and_solve_for_mgdaplus_1(vec, epsilon, lambda0):

    P = np.dot(vec, vec.T)

    n = P.shape[0]

    q = np.array([[0] for i in range(n)])

    A = np.ones(n).T
    b = np.array([1])

    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    sol = quadprog(P, q, G, h, A, b).reshape(-1)

    return sol, 1


def setup_qp_and_solve_for_FairMOO(grads, g_fair):

    P = np.dot(grads, grads.T)

    n = P.shape[0]
    q = np.zeros(n)

    G = np.zeros((n+1, n))
    h = np.zeros(n+1)
    for i in range(n):
        G[i][i] = -1
    G[n] = -grads @ g_fair

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag


def get_d_moomtl_d(grads, device):

    vec = grads
    sol, optimal_flag = setup_qp_and_solve(vec, device)

    d = torch.matmul(sol, grads)

    descent_flag = 1
    c = - (grads @ d)

    if not torch.all(c <= 1e-6):
        descent_flag = 0

    return d, optimal_flag, descent_flag

def get_d_FairMOO(grads, device, g_fair):
    sol, optimal_flag = setup_qp_and_solve_for_FairMOO(grads.cpu().detach().numpy(), g_fair.cpu().detach().numpy())
    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ grads
    return d, sol

def get_d_mgdaplus_d(grads, device, epsilon, lambda0):

    vec = grads
    sol, optimal_flag = setup_qp_and_solve_for_mgdaplus_1(
        vec.cpu().detach().numpy(), epsilon, lambda0)

    sol = torch.from_numpy(sol).float().to(device) 
    d = sol @ grads 

    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-6): 
        descent_flag = 0

    return d, sol, descent_flag

def check_constraints(value, ref_vec, prefer_vec):

    w = ref_vec - prefer_vec

    gx = torch.matmul(w, value/torch.norm(value))
    idx = gx > 0
    return torch.sum(idx), idx

def project(a, b):
    return a @ b / torch.norm(b)**2 * b

def solve_d(Q, g, value, device):
    L = value.cpu().detach().numpy()
    QTg = Q @ g.T
    QTg = QTg.cpu().detach().numpy()
    gTg = g @ g.T
    gTg = gTg.cpu().detach().numpy()

    def fun(x):
        return np.sum((gTg @ x - L)**2)

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x - 1e-10},
            {'type': 'ineq', 'fun': lambda x: QTg @ x}
            )

    x0 = np.random.rand(g.shape[0])
    x0 = x0 / np.sum(x0)
    res = minimize(fun, x0, method='SLSQP', constraints=cons)
    lam = res.x
    lam = torch.from_numpy(lam).float().to(device)
    d = lam @ g
    return d

def get_fedmdfg_d(grads, value, add_grads, alpha, fair_guidance_vec, force_active, device):
    fair_grad = None
    value_norm = torch.norm(value)
    norm_values = value / value_norm
    fair_guidance_vec /= torch.norm(fair_guidance_vec)
    cos = float(norm_values @ fair_guidance_vec)
    cos = min(1, cos)
    cos = max(-1, cos)
    bias = np.arccos(cos) / np.pi * 180
    pref_active_flag = (bias > alpha) | force_active
    norm_vec = torch.norm(grads, dim=1)
    indices = list(range(len(norm_vec)))
    grads = norm_vec[indices].reshape(-1, 1) * grads / (norm_vec + 1e-6).reshape(-1, 1)
    if not pref_active_flag:
        vec = grads
        pref_active_flag = 0
    else:
        pref_active_flag = 1
        h_vec = (fair_guidance_vec @ norm_values * norm_values - fair_guidance_vec).reshape(1, -1)
        h_vec /= torch.norm(h_vec)
        fair_grad = h_vec @ grads
        vec = torch.cat((grads, fair_grad))
    if add_grads is not None:
        norm_vec = torch.norm(add_grads, dim=1)
        indices = list(range(len(norm_vec)))
        random.shuffle(indices)
        add_grads = norm_vec[indices].reshape(-1, 1) * add_grads / (norm_vec + 1e-6).reshape(-1, 1)
        vec = torch.vstack([vec, add_grads])
    sol, _ = setup_qp_and_solve(vec.cpu().detach().numpy()) #sol, _ = setup_qp_and_solve(vec, device)
    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ vec
    return d, vec, pref_active_flag, fair_grad

def Gram_Schmidt(grads, loss, pow):
    num_vectors, dim = grads.shape
    orthogonal = torch.zeros_like(grads, dtype=torch.float64)

    for i in range(num_vectors):
        vec = grads[i].double().clone()
        numerator = grads[i].double().clone()
        denominator = loss[i] ** pow
        for j in range(i):
            numerator -= torch.dot(orthogonal[j], vec) / torch.dot(orthogonal[j], orthogonal[j]) * orthogonal[j]
            denominator -= torch.dot(orthogonal[j], vec) / torch.dot(orthogonal[j], orthogonal[j])
        orthogonal[i] = numerator / denominator
    return orthogonal.to(grads.dtype)

def get_d_adafed(grads):
    total = 0
    for grad in grads:
        total += 1.0 / torch.norm(grad) ** 2
    d = torch.zeros_like(grads[0])
    for grad in grads:
        lamb = 1.0 / (torch.norm(grad) ** 2 * total)
        d += lamb * grad
    return d