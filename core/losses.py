import torch
import torch.nn.functional as F

def loss_p(springback, prediction, coordinate_weights):
    loss = 0
    for i in range(3):
        loss += torch.mean(torch.norm(springback[:, i] - prediction[:, i], 2, 1)) * coordinate_weights[i]
    return loss

def loss_r(section, recover_section):
    return F.mse_loss(section, recover_section)