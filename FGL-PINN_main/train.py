import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pywt
import numpy as np

def loss(f, target=None):
    """
    Robust loss function for PINNs:
    - If target is None: returns MSE with zero.
    - If target is scalar: broadcast to f's shape.
    - If target is Tensor: compute standard MSE, handling shape mismatches.
    """
    if target is None:
        return torch.mean(torch.square(f))
    
    # If target is scalar (int/float), convert to tensor
    if isinstance(target, (int, float)):
        target = torch.tensor(target, dtype=f.dtype, device=f.device)
    
    if isinstance(target, torch.Tensor):
        # Ensure target has compatible shape
        if target.shape != f.shape:
            target = target.squeeze()
            if target.dim() == 0:  # Scalar case
                target = target.expand_as(f)
            else:
                if target.dim() == 1:
                    target = target.view(-1, 1)
                if target.shape[0] != f.shape[0]:
                    raise ValueError(f"Target size {target.shape} does not match prediction size {f.shape} at batch dimension")
                if target.shape[1:] != f.shape[1:]:
                    target = target.view(f.shape)
        return nn.MSELoss()(f, target)
    
    raise ValueError("Unsupported target type in loss computation.")

def grf_loss(pred, true, num_features=100, sigma=1.0):
    if pred.shape != true.shape:
        true = true.view(pred.shape)

    # Ensure input is 2D: [B, 1]
    pred = pred.view(-1, 1)
    true = true.view(-1, 1)

    # Generate random weights for Gaussian features
    device = pred.device
    W = torch.randn(1, num_features, device=device) * sigma  # Random weights [1, D]
    b = torch.rand(num_features, device=device) * 2 * torch.pi  # Random biases [D]

    # Project inputs to random feature space
    pred_features = torch.cos(pred @ W + b)  # [B, D]
    true_features = torch.cos(true @ W + b)  # [B, D]

    # Compute MSE in the feature space
    return F.mse_loss(pred_features, true_features)

def compute_laplacian(points, net, spatial_dims=[1, 2, 3]):
    """
    Compute Laplacian (sum of second derivatives) with respect to spatial dimensions.
    Args:
        points: Tensor of shape [B, 4], where columns are [t, x, y, z]
        net: Neural network model
        spatial_dims: List of column indices for spatial dimensions (default: [1, 2, 3] for x, y, z)
    Returns:
        Laplacian term: Tensor of shape [B, output_dim]
    """
    points = points.requires_grad_(True)  # Ensure points require gradients
    u = net(points)
    laplacian = 0
    for dim in spatial_dims:
        grad_u = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, dim:dim+1]
        grad_grad_u = torch.autograd.grad(grad_u, points, grad_outputs=torch.ones_like(grad_u), create_graph=True)[0][:, dim:dim+1]
        laplacian += grad_grad_u
    return laplacian

def train(net, PDE, BC, point_sets, flags, iterations=50000, lr=5e-4, info_num=100,
          test_in=None, test_out=None, w=[1., 1., 1., 1.], w_pde=1.0, w_grf=1.0,
          w_laplacian=1e-4, inv_params=[], num_features=100, sigma=1.0):

    if not inv_params:
        params = net.parameters()
    else:
        params = list(net.parameters()) + inv_params

    optimizer = torch.optim.Adam(params, lr=lr)

    # Precompute point counts for normalization
    n_bc = n_ic = n_PDE = n_data = 0
    for points, flag in zip(point_sets, flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_PDE += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]

    start_time = time.time()

    l_history = []
    err_history = [] if test_in is not None else None
    pde_loss_history = []
    bc_loss_history = []
    ic_loss_history = []
    data_loss_history = []
    grf_loss_history = []
    laplacian_loss_history = []

    for epoch in range(iterations):
        optimizer.zero_grad()
        l_BC = l_IC = l_PDE = l_data = l_grf = l_laplacian = 0

        for points, flag in zip(point_sets, flags):
            points = points.to(device=next(net.parameters()).device).requires_grad_(True)  # Ensure gradients for PDE and Laplacian
            if flag[0] == 'BC':
                f = BC(points[:, 0:1], points[:, 1:2], points[:, 2:3], points[:, 3:4], net, flag[1])
                target = torch.zeros_like(f)
                l_BC += loss(f, target) * points.shape[0] / n_bc 
            elif flag[0] == 'IC':
                pred = net(points)
                target = flag[1]
                l_IC += loss(pred, target) * points.shape[0] / n_ic
            elif flag[0] == 'data':
                pred = net(points)
                if isinstance(flag[1], torch.Tensor):
                    expected_shape = (points.shape[0], pred.shape[1])
                    if flag[1].shape != expected_shape:
                        flag[1] = flag[1].view(expected_shape) if flag[1].numel() == points.shape[0] * pred.shape[1] else \
                            torch.full(expected_shape, flag[1].item(), dtype=pred.dtype, device=pred.device)
                l_data += w[3] * loss(pred, flag[1]) * points.shape[0] / n_data
                if w_grf > 0:
                    l_grf += w_grf * grf_loss(pred, flag[1], num_features=num_features, sigma=sigma) * points.shape[0] / n_data
            elif flag[0] == 'domain':
                f = PDE(points[:, 0:1], points[:, 1:2], points[:, 2:3], points[:, 3:4], net)
                l_PDE += loss(f) * points.shape[0] / n_PDE
                l_laplacian += w_laplacian * loss(compute_laplacian(points, net)) * points.shape[0] / n_PDE

        # Compute physics loss with Laplacian
        l_physics = w[1] * l_BC + w[3] * l_IC + w[0] * l_PDE + l_laplacian

        # Total loss
        if n_data == 0:
            cost = l_physics
            l_history.append([cost.item(), l_BC.item(), l_IC.item(), l_PDE.item(), l_laplacian.item()])
        else:
            cost = w[2] * l_data + l_physics + l_grf
            l_history.append([cost.item(), l_BC.item(), l_IC.item(), l_PDE.item(), l_data.item(), l_grf.item(), l_laplacian.item()])

        # Append to individual loss histories
        pde_loss_history.append(l_PDE.item())
        bc_loss_history.append(l_BC.item())
        ic_loss_history.append(l_IC.item())
        data_loss_history.append(l_data.item())
        grf_loss_history.append(l_grf.item())
        laplacian_loss_history.append(l_laplacian.item())

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        cost.backward()

        # Compute GradNorm (after backward)
        grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None)).item()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()

        # Log gradient norm for debugging
        if epoch % info_num == 0:
            elapsed = time.time() - start_time
            if test_in is not None:
                T_pred = net(test_in)
                Test_err = loss(T_pred, test_out)
                err_history.append(Test_err.item())
                if n_data == 0:
                    print(f'It: {epoch}, Loss: {cost:.3e}, BC: {l_BC:.3e}, IC: {l_IC:.3e}, PDE: {l_PDE:.3e}, Laplacian: {l_laplacian:.3e}, Test: {Test_err:.3e}, GradNorm: {grad_norm:.3e}, Time: {elapsed:.2f}')
                else:
                    print(f'It: {epoch}, Loss: {cost:.3e}, BC: {l_BC:.3e}, IC: {l_IC:.3e}, PDE: {l_PDE:.3e}, Data: {l_data:.3e}, Wavelet: {l_grf:.3e}, Laplacian: {l_laplacian:.3e}, Test: {Test_err:.3e}, GradNorm: {grad_norm:.3e}, Time: {elapsed:.2f}')
            else:
                if n_data == 0:
                    print(f'It: {epoch}, Loss: {cost:.3e}, BC: {l_BC:.3e}, IC: {l_IC:.3e}, PDE: {l_PDE:.3e}, Laplacian: {l_laplacian:.3e}, GradNorm: {grad_norm:.3e}, Time: {elapsed:.2f}')
                else:
                    print(f'It: {epoch}, Loss: {cost:.3e}, BC: {l_BC:.3e}, IC: {l_IC:.3e}, PDE: {l_PDE:.3e}, Data: {l_data:.3e}, Wavelet: {l_grf:.3e}, Laplacian: {l_laplacian:.3e}, GradNorm: {grad_norm:.3e}, Time: {elapsed:.2f}')
            start_time = time.time()

            if inv_params:
                for value in inv_params:
                    print(f'Inv param: {value.item():.3e}')

    return l_history, err_history, pde_loss_history, bc_loss_history, ic_loss_history, data_loss_history, grf_loss_history, laplacian_loss_history