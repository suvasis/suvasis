# adopted from https://github.com/posgraph/coupe.PVDNet/blob/main/configs/config_PVDNet.py
from configs.config import get_config as main_config
from configs.config import log_config, print_config
import math
import torch
import numpy as np

def get_config(project = '', mode = '', config = '', data = '', LRS = '', batch_size = 8):

    ## GLOBAL
    config = main_config(project, mode, config, data, LRS, batch_size)

    ## LOCAL
    # tarining
    actual_batch_size = config.batch_size * torch.cuda.device_count()
    config.lr_init = 1e-4
    config.trainer = 'trainer'

    # networks
    config.network_BIMNet = 'liteFlowNet'
    #config.network = 'PVDNet'
    #config.network_PVDNet = config.network
    #config.network_BIMNet = 'liteFlowNet3'
    config.network = 'PVDNet'
    config.network_PVDNet = config.network
    #config.network_PVDNet = config.network.BIMNet

    config.PV_ksize = 5
    config.fix_BIMNet = True

    config.wi = 1.0 # weight init

    ## data
    config.frame_itr_num = 13
    config.frame_num = 3

    config.refine_val = 4 # mod crop value for PVDNet

    ## training schedule
    if config.data == 'nah':
        total_frame_num = int(6309/3)
        video_num = 22
    elif config.data == 'DVD':
        total_frame_num = int(11416/2)
        video_num = 61

    config.total_itr = 600000
    IpE = math.ceil((len(list(range(0, total_frame_num - (config.frame_itr_num-1), config.frame_itr_num)))) / actual_batch_size) * config.frame_itr_num
    #our_epoch = math.ceil(config.total_itr / IpE)

    if config.LRS == 'LD':
        # lr_decay
        config.decay_period = [400000]
        config.decay_rate = 0.25
        config.warmup_itr = -1
    elif config.LRS == 'CA':
        # Cosine Anealing
        config.warmup_itr = -1
        config.T_period = [50000, 100000, 150000, 150000, 150000]
        config.restarts = [50000, 150000, 300000, 450000]
        config.restart_weights = [1, 1, 1, 1]
        config.eta_min = 1e-7

    return config
