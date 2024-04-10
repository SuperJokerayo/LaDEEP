import torch
import torch.nn.functional as F

def loss_p(springback, prediction, coordinate_weights):
    loss = 0
    for i in range(3):
        loss += torch.mean(torch.norm(springback[:, i] - prediction[:, i], 2, 1)) * coordinate_weights[i]
    return loss

def loss_r(section, recovery, kl_weight = 0.00025):
    recover_section, mean, logvar = recovery
    recons_loss = F.mse_loss(section, recover_section)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss