import os
import logging
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.data_loader import LaDEEP_DataLoader
from core.ladeep import LaDEEP

from utils.tensorboard_logging import Tensorboard_Logging
from utils.diff_recover import diff_recover

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

def train(train_dataloader, eval_dataloader, device, exp_name, tl_writer):

    train_datas = DataLoader(
        train_dataloader,
        batch_size = train_batch_size, shuffle = True, num_workers = num_workers
    )

    eval_datas = DataLoader(
        eval_dataloader,
        batch_size = eval_batch_size, num_workers = num_workers
    )

    coordinate_weights = [train_dataloader.scale_factor[i] - train_dataloader.scale_factor[i + 3] for i in range(3)]

    checkpoint_save_path = f"./checkpoints/{exp_name}"
    if not os.path.exists(checkpoint_save_path):
        os.mkdir(checkpoint_save_path)

    # epochs = 600
    # lr_p, lr_r = 0.001, 0.001
    # weight_decay_p, weight_decay_r = 5e-5, 5e-5
    # torch.cuda.manual_seed(3407)

    torch.cuda.manual_seed(seed)

    net = LaDEEP().to(device)

    params_p = {n: p for n, p in net.named_parameters() if 'cse' not in n and 'csr' not in n}
    params_r = {n: p for n, p in net.named_parameters() if 'cse' in n or 'csr' in n}
    
    optimizer_p = optim.Adam(
        filter(lambda p: p.requires_grad, params_p.values()),
        lr = lr_p, 
        weight_decay = weight_decay_p
    )

    optimizer_r = optim.Adam(
        filter(lambda p: p.requires_grad, params_r.values()),
        lr = lr_r,
        weight_decay = weight_decay_r
    )

    scheduler_p = CosineAnnealingLR(
        optimizer_p, 
        epochs, 
        eta_min = lr_p / lr_p_decay_min
    )

    scheduler_r = CosineAnnealingLR(
        optimizer_r,
        epochs,
        eta_min = lr_r / lr_r_decay_min
    )

    len_train_datas = len(train_datas)
    len_eval_datas = len(eval_datas)

    min_train_loss_p = 1 << 30
    min_train_loss_r = 1 << 30
    min_eval_loss_p = 1 << 30
    min_eval_loss_r = 1 << 30

    for epoch in tqdm(range(epochs)):
        net = net.train()
        
        mean_train_loss_p = 0
        mean_train_loss_r = 0
        for i, train_data in enumerate(train_datas):
            strip, mould, section, params, springback = list(map(lambda x: x.float().to(device), train_data))
            prediction, recovery = net(strip, mould, section, params)
            train_loss_p = loss_p(springback, prediction, coordinate_weights)
            train_loss_r = loss_r(section, recovery)

            mean_train_loss_p += train_loss_p.data
            mean_train_loss_r += train_loss_r.data

            tl_writer.write_2d_figure("train/train_loss_p", train_loss_p.data, epoch * len_train_datas + i)
            tl_writer.write_2d_figure("train/train_loss_r", train_loss_r.data, epoch * len_train_datas + i)

            min_train_loss_p = min(train_loss_p.data, min_train_loss_p)
            min_train_loss_r = min(train_loss_r.data, min_train_loss_r)

            tl_writer.write_2d_figure("train/min_train_loss_p", min_train_loss_p.data, epoch * len_train_datas + i)
            tl_writer.write_2d_figure("train/min_train_loss_r", min_train_loss_r.data, epoch * len_train_datas + i)
            
            optimizer_r.zero_grad()
            optimizer_p.zero_grad()
            train_loss_p.backward(retain_graph = True)
            train_loss_r.backward()
            optimizer_p.step()
            optimizer_r.step()

            
        scheduler_p.step()
        scheduler_r.step()

        mean_train_loss_p /= len_train_datas
        mean_train_loss_r /= len_train_datas

        tl_writer.write_2d_figure("train/mean_train_loss_p", mean_train_loss_p, epoch)
        tl_writer.write_2d_figure("train/mean_train_loss_r", mean_train_loss_r, epoch)


        net = net.eval()
        with torch.no_grad():
            mean_eval_loss_p = 0
            mean_eval_loss_r = 0
            for i, eval_data in enumerate(eval_datas):
                strip, mould, section, params, springback = list(map(lambda x: x.float().to(device), eval_data))
                prediction, recovery = net(strip, mould, section, params)
                eval_loss_p = loss_p(springback, prediction, coordinate_weights)
                eval_loss_r = loss_r(section, recovery)

                tl_writer.write_2d_figure("eval/eval_loss_p", eval_loss_p, epoch * len_eval_datas + i)
                tl_writer.write_2d_figure("eval/eval_loss_r", eval_loss_r, epoch * len_eval_datas + i)

                mean_eval_loss_p += eval_loss_p.data
                mean_eval_loss_r += eval_loss_r.data
            mean_eval_loss_p /= len_eval_datas
            mean_eval_loss_r /= len_eval_datas

            tl_writer.write_2d_figure("eval/mean_eval_loss_p", mean_eval_loss_p, epoch)
            tl_writer.write_2d_figure("eval/mean_eval_loss_r", mean_eval_loss_r, epoch)
            
            if mean_eval_loss_p < min_eval_loss_p:
                save_params_p = {n: p for n, p in net.state_dict().items() if 'cse' not in n and 'csr' not in n}
                torch.save(
                    save_params_p,
                    os.path.join(checkpoint_save_path, "best_p_model.pth")
                )
                min_eval_loss_p = mean_eval_loss_p
                logging.info(f"checkpoint p saved at {epoch} with loss {min_eval_loss_p}")

            if mean_eval_loss_r < min_eval_loss_r:
                save_params_r = {n: p for n, p in net.state_dict().items() if 'cse' in n or 'csr' in n}
                torch.save(
                    save_params_r,
                    os.path.join(checkpoint_save_path, "best_r_model.pth")
                )
                min_eval_loss_r = mean_eval_loss_r
                logging.info(f"checkpoint r saved at {epoch} with loss {min_eval_loss_r}")
    logging.info("training finished")
    logging.info(f"min_eval_loss_p : {min_eval_loss_p}")
    logging.info(f"min_eval_loss_r : {min_eval_loss_r}")

def test(data_path, test_dataloader, device, exp_name, tl_writer):

    coordinate_weights = [test_dataloader.scale_factor[i] - test_dataloader.scale_factor[i + 3] for i in range(3)]

    test_datas = DataLoader(
        test_dataloader, batch_size = test_batch_size, num_workers = num_workers
    )

    data_path = os.path.join(data_path, f"prediction_results/{exp_name}")
    prediction_line_path = os.path.join(data_path, "prediction_line")
    prediction_section_path = os.path.join(data_path, "prediction_section")

    if not os.path.exists(prediction_line_path):
        os.makedirs(prediction_line_path)
    if not os.path.exists(prediction_section_path):
        os.makedirs(prediction_section_path)


    net = LaDEEP().to(device)
    checkpoint_p_path = f"./checkpoints/{exp_name}/best_model_p.pth"
    checkpoint_r_path = f"./checkpoints/{exp_name}/best_model_r.pth"

    parameters_p = torch.load(checkpoint_p_path, map_location = device)
    parameters_r = torch.load(checkpoint_r_path, map_location = device)


    net_parameters = net.state_dict()
    print(len(net_parameters.keys()))
    print(len(parameters_p.keys()) + len(parameters_r.keys()))

    for key in net.named_parameters().keys():
        if key not in parameters_p.keys() and key not in parameters_r.keys():
            print(key)
    
    net_parameters.update(parameters_p)
    net_parameters.update(parameters_r)
    net.load_state_dict(net_parameters)

    len_test_datas = len(test_datas)
    
    lines, sections = [], []
    net = net.eval()
    with torch.no_grad():
        mean_test_loss_p = 0
        mean_test_loss_r = 0

        for i, test_data in enumerate(test_datas):
            strip, mould, section, params, springback = list(map(lambda x: x.float().to(device), test_data))
            prediction, recovery = net(strip, mould, section, params)

            test_loss_p = loss_p(springback, prediction, coordinate_weights)
            test_loss_r = loss_r(section, recovery)

            mean_test_loss_p += test_loss_p.data
            mean_test_loss_r += test_loss_r.data
            lines.append(prediction.cpu().detach().numpy())
            sections.append(recovery[0].cpu().detach().numpy())


        mean_test_loss_p /= len_test_datas
        mean_test_loss_r /= len_test_datas

        logging.info(f"mean_test_loss_p: {mean_test_loss_p}  mean_test_loss_r{mean_test_loss_r}")
    
    prediction_line, recovery_section = lines[0], sections[0]
    for i in range(1, len(lines)):
        prediction_line = np.concatenate((prediction_line, lines[i]), axis = 0)
        recovery_section = np.concatenate((recovery_section, sections[i]), axis = 0)    
    mean_test_dist_p = 0

    for i in range(3):
        prediction_line[:, i, :] = prediction_line[:, i, :] * (coordinate_weights[i] - coordinate_weights[i + 3]) + coordinate_weights[i + 3]


    mean_test_dist_last_point, max_test_dist_last_point, min_test_dist_last_point = 0, 0, 1 << 30
    
    max_sample_mean_test_dist_p, min_sample_mean_test_dist_p = 0, 1 << 30

    for j in range(prediction_line.shape[0]):
        pred = diff_recover(prediction_line[j, :, :].T)
        with open(test_dataloader.springback_line_paths[j]) as f:
            points = np.array(
                        list(
                            map(
                                lambda x: list(
                                    map(
                                        lambda y: float(y), 
                                        x.split()
                                    )
                                ), 
                                f.readlines()
                            )
                        )
                    )
        distance = np.sqrt(np.sum((pred - points) ** 2, axis = 1))
        test_dist_last_point = np.sqrt(np.sum((pred[-1] - points[-1]) ** 2))
        mean_test_dist_last_point += test_dist_last_point
        max_test_dist_last_point = max(max_test_dist_last_point, test_dist_last_point)
        min_test_dist_last_point = min(min_test_dist_last_point, test_dist_last_point)

        sample_mean_test_dist_p = np.mean(distance)
        max_sample_mean_test_dist_p = max(max_sample_mean_test_dist_p, sample_mean_test_dist_p)
        min_sample_mean_test_dist_p = min(min_sample_mean_test_dist_p, sample_mean_test_dist_p) 

        tl_writer.write_2d_figure("test/sample_mean_test_dist_p", sample_mean_test_dist_p, j)
        tl_writer.write_2d_figure("test/main_max_test_dist", np.max(distance), j)

        mean_test_dist_p += sample_mean_test_dist_p

        type_idx = test_dataloader.mould_line_paths[j].find("type")
        type_id = test_dataloader.mould_line_paths[j][type_idx : type_idx + 6]
        type_path = os.path.join(prediction_line_path, type_id)
        if not os.path.exists(type_path):
            os.mkdir(type_path)
        file_path = os.path.join(type_path, test_dataloader.mould_line_paths[j][-8 : ])
        with open(file_path, 'w', encoding = "utf8") as w:
            for k in range(pred.shape[0]):
                w.write(f"{pred[k][0]} {pred[k][1]} {pred[k][2]}\n")
        type_path = os.path.join(prediction_section_path, type_id)
        if not os.path.exists(type_path):
            os.mkdir(type_path)
        
        file_path = os.path.join(type_path, f"{test_dataloader.mould_line_paths[j][-8 : -3]}.jpg")
        sec = recovery_section[j, :, :, :]* 255
        sec = np.transpose(sec, (1, 2, 0)).astype(np.uint8).squeeze() 
        sec = Image.fromarray(sec)
        sec.save(file_path)

    mean_test_dist_p /= prediction_line.shape[0]
    mean_test_dist_last_point /= prediction_line.shape[0]
    logging.info(f"mean_test_dist_last_point: {mean_test_dist_last_point}")
    logging.info(f"max_test_dist_last_point: {max_test_dist_last_point}")
    logging.info(f"min_test_dist_last_point: {min_test_dist_last_point}")
    logging.info(f"mean_test_dist_p: {mean_test_dist_p}")
    logging.info(f"max_sample_mean_test_dist_p: {max_sample_mean_test_dist_p}")
    logging.info(f"min_sample_mean_test_dist_p: {min_sample_mean_test_dist_p}")

def main():
    mode = "train"
    # mode_id = 1
    exp_name = f"{mode}_{mode_id}"

    # device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    logs_dir = f"./logs/{exp_name}/"
    if os.path.exists(logs_dir):
        exit("logs dir exists, please set a new dir!")

    tl_writer = Tensorboard_Logging(logs_dir)

    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt = '%m-%d %H:%M',
                        filename = os.path.join(logs_dir, "log.log"),
                        filemode = 'w')

    # data_path = "/mnt/data/tao_shilong/DataBase/e2e_model_data"

    strip_path = os.path.join(data_path, "original_strip_line")
    mould_path =  os.path.join(data_path, "mould_line")
    section_path = os.path.join(data_path, "strip_section_tiff")
    params_path = os.path.join(data_path, "stretch_bending_params")
    springback_path = os.path.join(data_path, "springback_strip_line")
    
    if mode == "train":
        train_dataloader = LaDEEP_DataLoader(
            strip_path = strip_path,
            mould_path = mould_path,
            section_path = section_path,
            params_path = params_path,
            springback_path = springback_path,
            mode = "train"
        )
        eval_dataloader = LaDEEP_DataLoader(
            strip_path = strip_path,
            mould_path = mould_path,
            section_path = section_path,
            params_path = params_path,
            springback_path = springback_path,
            mode = "eval"
        )
        train(train_dataloader, eval_dataloader, device, exp_name, tl_writer)
    elif mode == "test":
        test_dataloader = LaDEEP_DataLoader(
            strip_path = strip_path,
            mould_path = mould_path,
            section_path = section_path,
            params_path = params_path,
            springback_path = springback_path,
            mode = "test"
        )

        test(data_path, test_dataloader, device, exp_name, tl_writer)

    tl_writer.writer_close()

if __name__ == "__main__":
    from configparser import ConfigParser
    config = ConfigParser()
    config.read("./config.ini", encoding = "utf-8")
    mode = config.get("settings", "mode")
    mode_id = config.getint("settings", "mode_id")
    device_id = config.getint("settings", "device_id")
    data_path = config.get("settings", "data_path")
    num_workers = config.getint("settings", "num_workers")

    train_batch_size = config.getint("hyper_parameters", "train_batch_size")
    eval_batch_size = config.getint("hyper_parameters", "eval_batch_size")
    test_batch_size = config.getint("hyper_parameters", "test_batch_size")
    epochs = config.getint("hyper_parameters", "epochs")
    lr_p = config.getfloat("hyper_parameters", "lr_p")
    lr_r = config.getfloat("hyper_parameters", "lr_r")
    weight_decay_p = config.getfloat("hyper_parameters", "weight_decay_p")
    weight_decay_r = config.getfloat("hyper_parameters", "weight_decay_r")
    seed = config.getint("hyper_parameters", "seed")
    lr_p_decay_min = config.getfloat("hyper_parameters", "lr_p_decay_min")
    lr_r_decay_min = config.getfloat("hyper_parameters", "lr_r_decay_min")

    main()