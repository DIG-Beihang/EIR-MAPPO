import numpy as np
import torch
import torch.nn as nn


def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        if hessian is None:
            continue
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_divergence(obs, rnn_states, action, masks, available_actions, active_masks, new_actor, old_actor):
    _, _, new_dist = new_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks)
    with torch.no_grad():
        _, _, old_dist = old_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks)
    return torch.distributions.kl.kl_divergence(old_dist, new_dist)

# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py


def conjugate_gradient(actor, obs, rnn_states, action, masks, available_actions, active_masks, b, nsteps, device, residual_tol=1e-10):
    x = torch.zeros(b.size()).to(device=device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = fisher_vector_product(
            actor, obs, rnn_states, action, masks, available_actions, active_masks, p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def fisher_vector_product(actor, obs, rnn_states, action, masks, available_actions, active_masks, p):
    with torch.backends.cudnn.flags(enabled=False):
        p.detach()
        kl = kl_divergence(obs, rnn_states, action, masks,
                           available_actions, active_masks, new_actor=actor, old_actor=actor)
        kl = kl.mean()
        kl_grad = torch.autograd.grad(
            kl, actor.parameters(), create_graph=True, allow_unused=True)
        kl_grad = flat_grad(kl_grad)  # check kl_grad == 0
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(
            kl_grad_p, actor.parameters(), allow_unused=True)
        kl_hessian_p = flat_hessian(kl_hessian_p)
        return kl_hessian_p + 0.1 * p
