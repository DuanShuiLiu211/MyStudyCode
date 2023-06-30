# encoding:utf-8
import sys
import torch


# TODO add muti-device setting
def devices(config, logger):
    device_type = config['device_type']
    if device_type == 'cuda':
        cuda_idx = int(config['cuda_idx'])
        if torch.cuda.is_available():
            gpu_num = torch.cuda.device_count()
            if cuda_idx < gpu_num:
                dev = torch.device('cuda:{}'.format(config['cuda_idx']))
                logger.info(f'the model will train on {dev}')
            else:
                dev = torch.device('cuda:0')
                logger.info(f'the model will train on {dev}')
        else:
            dev = torch.device('cpu')
            logger.info(f'the model will train on {dev}')
    elif device_type == 'mps':
        if torch.backends.mps.is_available():
            dev = torch.device('mps')
            logger.info(f'the model will train on {dev}')
        else:
            dev = torch.device('cpu')
            logger.info(f'the model will train on {dev}')
    else:
        if device_type == 'cpu':
            dev = torch.device('cpu')
            logger.info(f'the model will train on {dev}')
        else:
            logger.warning(f'the device_type:{device_type} is invalid')
            sys.exit()
    return dev
