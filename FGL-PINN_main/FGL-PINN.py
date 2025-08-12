import os
import time
import platform
import psutil
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch import nn
from fvcore.nn import FlopCountAnalysis, parameter_count
from thop import profile
from torchinfo import summary
from skopt import gp_minimize
from skopt.space import Real
import pyvista as pv
from model2 import FNN
from util import *
from train18 import *

torch.manual_seed(0)


def output_transform(X):
    X = T_range*nn.Softplus()(X)+ T_ref
    return X


def input_transform(X):
    X = 2.*(X-X_min)/(X_max-X_min) - 1.
    return X


def PDE(x, y, z, t, net):
    X = torch.cat([x, y, z, t], dim=-1)
    T = net(X)
    T_t = grad(T, t, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_xx = grad(T_x, x, create_graph=True, grad_outputs=torch.ones_like(T_x))[0]
    T_y = grad(T, y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_yy = grad(T_y, y, create_graph=True, grad_outputs=torch.ones_like(T_y))[0]
    T_z = grad(T, z, create_graph=True, grad_outputs=torch.ones_like(T))[0]
    T_zz = grad(T_z, z, create_graph=True, grad_outputs=torch.ones_like(T_z))[0]

    conduction = k * (T_xx + T_yy + T_zz)
    convection =rho * Cp *U* (T_x+T_y+T_z)
    f = rho * Cp * T_t - conduction + convection
    return f
import numpy as np

def sampling_uniform(density, x_range, y_range, z_range=None, t_range=None, sample_type='domain', t=None):
    """
    Generates uniform sampling points in the specified range.
    """
    if z_range is None and t_range is None:
        # 2D sampling
        x = np.random.uniform(x_range[0], x_range[1], int(density))
        y = np.random.uniform(y_range[0], y_range[1], int(density))
        return np.stack([x, y], axis=-1), None
    elif t_range is None:
        # 3D sampling
        x = np.random.uniform(x_range[0], x_range[1], int(density))
        y = np.random.uniform(y_range[0], y_range[1], int(density))
        z = np.random.uniform(z_range[0], z_range[1], int(density))
        return np.stack([x, y, z], axis=-1), None
    else:
        # 4D sampling
        x = np.random.uniform(x_range[0], x_range[1], int(density))
        y = np.random.uniform(y_range[0], y_range[1], int(density))
        z = np.random.uniform(z_range[0], z_range[1], int(density))
        t = np.random.uniform(t_range[0], t_range[1], int(density))
        return np.stack([x, y, z, t], axis=-1), None
def generate_points(p=[], f=[]):
    t = np.linspace(x_min[3] + 0.04, x_max[3], 250)
    
    bound_x_neg, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='-x')
    bound_x_pos, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='+x')
    bound_y_neg, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='-y')
    bound_y_pos, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='+y')
    bound_z_neg, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='-z')
    bound_z_pos, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [t.min(), t.max()], sample_type='+z')

    melt_pool_x_range = [x_min[0], x_max[0]]
    melt_pool_y_range = [x_min[1], x_max[1]]
    melt_pool_z_range = [x_min[2], x_max[2]]
    melt_pool_t_range = [x_min[3], x_max[3]]

    melt_pool_pts, _ = sampling_uniform(4., melt_pool_x_range, melt_pool_y_range, melt_pool_z_range, melt_pool_t_range, 'melt_pool')

    domain_pts, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [x_min[3], x_max[3]], 'domain')
    init_pts, _ = sampling_uniform(2., [x_min[0], x_max[0]], [x_min[1], x_max[1]], [x_min[2], x_max[2]], [0, 0], 'domain')

    p.extend([
        torch.tensor(bound_x_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_x_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_y_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_y_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_z_neg, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(bound_z_pos, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(init_pts, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(domain_pts, requires_grad=True, dtype=torch.float).to(device),
        torch.tensor(melt_pool_pts, requires_grad=True, dtype=torch.float).to(device)
    ])
    
    f.extend([
        ['BC', '-x'], ['BC', '+x'], ['BC', '-y'], ['BC', '+y'], 
        ['BC', '-z'], ['BC', '+z'], ['IC', T_ref], ['domain'], ['melt_pool']
    ])
    
    return p, f
def load_data(p,f,filename,num):
    data = np.load(filename)
    if num!= 0:
        np.random.shuffle(data)
        data = data[0:num,:]
    p.extend([torch.tensor(data[:,0:4],requires_grad=True,dtype=torch.float).to(device)])
    f.extend([['data',torch.tensor(data[:,4:5],requires_grad=True,dtype=torch.float).to(device)]])
    return p,f


def BC(x, y, z, t, net, loc):
    X = torch.concat([x, y, z, t], axis=-1)
    T = net(X)
    if loc == '-x':
        T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return k * T_x - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4)
    if loc == '+x':
        T_x = grad(T, x, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return -k * T_x - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4)
    if loc == '-y':
        T_y = grad(T, y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return k * T_y - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4)
    if loc == '+y':
        T_y = grad(T, y, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return -k * T_y - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4)
    if loc == '-z':
        T_t = grad(T, t, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        return T_t
    if loc == '+z':
        T_z = grad(T, z, create_graph=True, grad_outputs=torch.ones_like(T))[0]
        q = 2 * P * eta / torch.pi / r**2 * torch.exp(-2 * (torch.square(x - 12 - v * t) + torch.square(y - 2.5)) / r**2) * (t <= t_end) * (t > 0)
        return -k * T_z - h * (T - T_ref) - Rboltz * emiss * (T**4 - T_ref**4) + q
if __name__ == '__main__':
    
    # augments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='GPU name')
    parser.add_argument('--output', type=str, default='bareplate', help='output filename')
    parser.add_argument('--T_ref', type=float, default=..., help='ambient temperature')
    parser.add_argument('--T_range', type=float, default=..., help='temperature range')
    parser.add_argument('--xmax', type=float, default=..., help='max x')
    parser.add_argument('--xmin', type=float, default=..., help='min x')
    parser.add_argument('--ymax', type=float, default=..., help='max y')
    parser.add_argument('--ymin', type=float, default=..., help='min y')
    parser.add_argument('--zmax', type=float, default=..., help='max z')
    parser.add_argument('--zmin', type=float, default=..., help='min z')
    parser.add_argument('--tmax', type=float, default=..., help='max t')
    parser.add_argument('--tmin', type=float, default=..., help='min t')
    parser.add_argument('--Cp', type=float, default=..., help='specific heat')
    parser.add_argument('--k', type=float, default=..., help='heat conductivity')
    parser.add_argument('--x0', type=float, default=..., help='toolpath origin x')
    parser.add_argument('--y0', type=float, default=..., help='toolpath origin y')
    parser.add_argument('--r', type=float, default=..., help='beam radius')
    parser.add_argument('--v', type=float, default=..., help='scan speed')
    parser.add_argument('--t_end', type=float, default=..., help='laser stop time')
    parser.add_argument('--h', type=float, default=..., help='convection coefficient')
    parser.add_argument('--eta', type=float, default=..., help='absorptivity')
    parser.add_argument('--P', type=float, default=..., help='laser power')
    parser.add_argument('--emiss', type=float, default=..., help='emissivity')
    parser.add_argument('--rho', type=float, default=..., help='rho')
    parser.add_argument('--iters', type=int, default=..., help='number of iters')
    parser.add_argument('--lr', type=float, default=..., help='learning rate')
    parser.add_argument('--data', type=str, default='X_train.npy', help='filename, default:None')
    parser.add_argument('--data_num', type=int, default= '...', help='number of training data used, 0 for all data')
    parser.add_argument('--calib_eta',type=bool, default = False, help='calibrate eta')
    parser.add_argument('--calib_material',type=bool, default = False, help='calibrate cp and k')
    parser.add_argument('--valid',type=str, default = 'X_valid.npy', help='validation data file')
    parser.add_argument('--pretrain',type=str, default = '...', help='pretrained model file')
    parser.add_argument('--U', type=float, default=..., help='convection velocity')
    args = parser.parse_args()
    
    ##############params 
    device = torch.device("cuda:0")

    # domain
    x_max = np.array([args.xmax, args.ymax, args.zmax, args.tmax])
    x_min = np.array([args.xmin, args.ymin, args.zmin, args.tmin])
    X_max = torch.tensor(x_max,dtype=torch.float).to(device)
    X_min = torch.tensor(x_min,dtype=torch.float).to(device)
    
    # laser params
    x0 = args.x0
    y0 = args.y0
    r = args.r
    v = args.v # speed
    t_end = args.t_end
    P = args.P # power
    eta = args.eta

    # T_ambient, and max T range
    T_ref = args.T_ref
    T_range = args.T_range

    # material params
    Cp = args.Cp
    k = args.k
    h = args.h
    U= args.U
    Rboltz = 5.6704e-14
    emiss = args.emiss
    rho = args.rho
    
    # valid data
    data = np.load(args.valid)
    test_in = torch.tensor(data[:,0:4],requires_grad=False,dtype=torch.float).to(device)
    test_out = torch.tensor(data[:,4:5],requires_grad=False,dtype=torch.float).to(device)
    lr = args.lr

    def bayesian_objective(params, net, PDE, BC, point_sets, flags, inv_params, test_in, test_out):
        lr = params[0]
        w_pde, w_bc, w_data, w_ic = params[1:5]
        w_grf = params[5]
        w_laplacian = params[6]

    # 使用少量迭代评估参数效果
        l_history, err_history, pde_loss_history, bc_loss_history, ic_loss_history, data_loss_history, grf_loss_history, laplacian_loss_history = train(
        net=net,
        PDE=PDE,
        BC=BC,
        point_sets=point_sets,
        flags=flags,
        iterations=100,  # 少量迭代以节省计算
        lr=lr,
        info_num=100,
        test_in=test_in,
        test_out=test_out,
        w=[w_pde, w_bc, w_data, w_ic],  # w[0]: PDE, w[1]: BC, w[2]: data, w[3]: IC
        w_grf=w_grf,
        w_laplacian=w_laplacian,
        inv_params=inv_params,
        num_features=100,
        sigma=1.0
    )

    # 提取损失：l_history = [cost, l_BC, l_IC, l_PDE, l_data, l_wavelet, l_laplacian]
        # Convert l_history to NumPy array for indexing
        l_history = np.array(l_history)
    # Extract losses: l_history columns are [cost, l_BC, l_IC, l_PDE, l_data, l_wavelet, l_laplacian]
        pde_loss = np.mean(l_history[:, 3])  # PDE loss mean
        bc_loss = np.mean(l_history[:, 1])   # BC loss mean
        ic_loss = np.mean(l_history[:, 2])   # IC loss mean
        data_loss = np.mean(l_history[:, 4]) # Data loss mean
        test_loss = np.mean(np.array(err_history))  # Test error mean

    # 计算加权总损失，强调数据和IC损失以加速其下降
        total_loss = w_pde * pde_loss + w_bc * bc_loss + w_ic * ic_loss + w_data * data_loss + test_loss

    # 正则化项
    # L2正则化惩罚学习率和权重
        l2_penalty = 1e-4 * (lr**2)
        l2_weight_penalty = 1e-4 * (w_pde**2 + w_bc**2 + w_data**2 + w_ic**2 + w_wavelet**2 + w_laplacian**2)

    # 惩罚不合理的参数范围
        penalty_lr = 0
        if lr > 1e-2:
           penalty_lr += (lr - 1e-2)**2
        if lr < 1e-5:
           penalty_lr += (lr - 1e-5)**2

        penalty_weights = 0
        for w in [w_pde, w_bc, w_data, w_ic, w_wavelet, w_laplacian]:
            if w > 100.0:
                 penalty_weights += (w - 100.0)**2
            if w < 1e-6:
                 penalty_weights += (w - 1e-6)**2

    # 惩罚PDE损失波动（标准差）
        pde_std = np.std(l_history[:, 3])
        fluctuation_penalty = 1e-2 * pde_std if pde_std > 1e-2 else 0

    # 总正则化损失
        regularization_loss = l2_penalty + l2_weight_penalty + penalty_lr + penalty_weights + fluctuation_penalty

    # 最终目标函数：损失 + 正则化
        final_loss = total_loss + regularization_loss

        return final_loss

# 定义搜索空间
    space = [
    Real(1e-5, 1e-2, name='lr'),           # 学习率
    Real(1e-4, 1.0, name='w_pde'),        # PDE权重
    Real(1e-4, 1.0, name='w_bc'),          # BC权重
    Real(1e-4, 1.0, name='w_data'),      # 数据权重
    Real(1e-4, 1.0, name='w_ic'),        # IC权重
    Real(1e-6, 1e-2, name='w_grf'),    # 高斯随机损失权重
    Real(1e-6, 1e-2, name='w_laplacian')   # 拉普拉斯权重
]

    # In main.py
    num_fourier_frequencies = 5  
    layers = [128, 128, 128, 1]  
   
    net = FNN(layers=layers, activation=nn.Tanh(), num_fourier_frequencies=num_fourier_frequencies, scale=1.0, in_tf=input_transform,out_tf=output_transform)

    net.to(device)
    point_sets, flags = generate_points([], [])  
    if args.data != 'None':
       point_sets, flags = load_data(point_sets, flags, args.data, args.data_num)

    inv_params = []
    if args.calib_eta:
        eta = torch.tensor(1e-5, requires_grad=True, device=device)
        inv_params.append(eta)
    if args.calib_material:
        Cp = torch.tensor(1e-5, requires_grad=True, device=device)
        inv_params.append(Cp)
        k = torch.tensor(1e-5, requires_grad=True, device=device)
        inv_params.append(k)

# 贝叶斯优化
    objective = partial(
    bayesian_objective,
    net=net,
    PDE=PDE,
    BC=BC,
    point_sets=point_sets,
    flags=flags,
    inv_params=inv_params,
    test_in=test_in,
    test_out=test_out
)

    print("开始贝叶斯优化...")
    res = gp_minimize(objective, space, n_calls=..., random_state=42) 

# 输出优化结果
    best_lr = res.x[0]
    best_weights = res.x[1:5]
    best_w_grf = res.x[5]
    best_w_laplacian = res.x[6]
    print(f"最佳学习率: {best_lr:.3e}")
    print(f"最佳权重 [w_pde, w_bc, w_data, w_ic]: {best_weights}")
    print(f"最佳高斯随机损失权重: {best_w_grf:.3e}")
    print(f"最佳拉普拉斯权重: {best_w_laplacian:.3e}")

# 使用优化后的参数进行最终训练
    l_history, err_history, pde_loss_history, bc_loss_history, ic_loss_history, data_loss_history, grf_loss_history, laplacian_loss_history = train(
    net=net,
    PDE=PDE,
    BC=BC,
    point_sets=point_sets,
    flags=flags,
    iterations=5000,
    lr=best_lr,
    info_num=100,
    test_in=test_in,
    test_out=test_out,
    w=best_weights,
    w_pde=1.0,
    w_grf=best_w_grf,
    w_laplacian=best_w_laplacian,
    inv_params=inv_params,
    num_features=100,
    sigma=1.0
)
    torch.save(net.state_dict(),'....pt')
    np.save('../pde_loss_history.npy', np.array(pde_loss_history))
    np.save('.../bc_loss_history.npy', np.array(bc_loss_history))
    np.save('.../ic_loss_history.npy', np.array(ic_loss_history))
    np.save('.../data_loss_history.npy', np.array(data_loss_history))
    np.save('.../grf_loss_history.npy', np.array(grf_loss_history))
    np.save('.../laplacian_loss_history.npy', np.array(laplacian_loss_history))
    np.save('.../loss_history.npy', np.array(l_history))
