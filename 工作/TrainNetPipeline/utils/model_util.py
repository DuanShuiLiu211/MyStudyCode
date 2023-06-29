# encoding:utf-8
import torch
import torch.nn as nn
import sys
import time
import timm
import copy
import gc
from thop import profile


def count_model_complexity(model, device, logger):
    model_copy = copy.deepcopy(model)
    input_size = [1, 3, 224, 224]
    inputs = torch.randn(input_size, device=device)
    time_start = time.time_ns()
    try:
        macs, params = profile(model_copy, inputs=(inputs, inputs))
    except TypeError:
        macs, params = profile(model_copy, inputs=(inputs,))
    time_end = time.time_ns()
    take_time = time_end - time_start
    
    blank = ' '
    logger.info('-' * 119)
    logger.info('|' + ' ' * 30 + 'weight name' + ' ' * 30 + '|' + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' + ' ' * 3 +
                'number' + ' ' * 3 + '|')
    logger.info('-' * 119)
    num_para = 0

    for index, (key, w_variable) in enumerate(model_copy.named_parameters()):
        if len(key) <= 69:
            key = key + (69 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank
        logger.info(f'{index}')
        logger.info('| {} | {} | {} |'.format(key, shape, str_num))

    mn = type(model_copy).__name__
    logger.info('-' * 119)
    logger.info(f'the {mn} macs = ' + str(macs / 1e9) + 'G')
    logger.info(f'the {mn} times = ' + str(take_time / 1e6) + 'ms')
    logger.info(f'the {mn} parameters calculate by 「thop_profile」 = ' + str(params/1e6) + 'M')
    logger.info(f'the {mn} parameters calculate by 「named_parameters」 = {num_para/1e6:.2f}M')
    logger.info('-' * 119)
    del model_copy
    gc.collect()


def load_train_verify_model(config, model, device, logger):
    logger.info(model)
    count_model_complexity(model, device, logger)
    load_mode = config['model_load_mode']
    model_weight = config['train_verify_model_weight'] 
    try:
        if load_mode == "checkpoint":
            new_weight_dict = torch.load(model_weight, map_location=device).state_dict()
        elif load_mode == "weight":
            new_weight_dict = torch.load(model_weight, map_location=device)
        else:
            new_weight_dict = {}
            logger.warning(f'the load model {load_mode} is exist, can not load model weight from {model_weight} for model {type(model).__name__}, the parameters of model is initialized by method in model set')
        old_weight_dict = model.state_dict()
        updated_weight_dict = {k: v for k, v in new_weight_dict.items() if k in old_weight_dict}
        old_weight_dict.update(updated_weight_dict)
        model.load_state_dict(old_weight_dict)
        new_params = len(new_weight_dict)
        old_params = len(old_weight_dict)
        matched_params = len(updated_weight_dict)
        logger.info(f'load model weight from {model_weight} for {type(model).__name__} model')
        logger.info(f'the new model params:{new_params}, old model params:{old_params}, matched params:{matched_params}')
    except FileNotFoundError:
        logger.warning(f'the load model {load_mode} is exist, can not load model weight from {model_weight} for model {type(model).__name__}, the parameters of model is initialized by method in model set')

    return model


def load_test_model(config, model, device, logger):
    logger.info(model)
    count_model_complexity(model, device, logger)
    load_mode = config['model_load_mode']
    model_weight = config['test_model_weight'] 
    try:
        if load_mode == "checkpoint":
            model = torch.load(model_weight, map_location=device)
            logger.info(f'load model checkpoint from {model_weight} for {type(model).__name__} model')
        elif load_mode == "weight":
            new_weight_dict = torch.load(model_weight, map_location=device)
            old_weight_dict = model.state_dict()
            updated_weight_dict = {k: v for k, v in new_weight_dict.items() if k in old_weight_dict}
            old_weight_dict.update(updated_weight_dict)
            model.load_state_dict(old_weight_dict)
            new_params = len(new_weight_dict)
            old_params = len(old_weight_dict)
            matched_params = len(updated_weight_dict)
            logger.info(f'load model weight from {model_weight} for {type(model).__name__} model')
            logger.info(f'the new model params:{new_params}, old model params:{old_params}, matched params:{matched_params}')
        else:
            logger.warning(f'the load model {load_mode} is exist, can not load model weight from {model_weight} for model {type(model).__name__}, the parameters of model is initialized by method in model set')
    except FileNotFoundError:
        logger.warning(f'the model weight {model_weight} is exist, can not load model weight from {model_weight} for model {type(model).__name__}, the parameters of model is initialized by method in model set')

    return model


def get_model(config, device, logger):
    # 选择需要使用的模型
    model_name = config['model_name']
    model_mode = config['model_mode']
    if model_name == 'resnet18':
        model = timm.create_model("resnet18.fb_ssl_yfcc100m_ft_in1k", pretrained=False, num_classes=3)
        model = model.to(device)
    elif model_name == 'resnet50':
        model = timm.create_model("resnet50.fb_ssl_yfcc100m_ft_in1k", pretrained=False, num_classes=3)
        model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=256, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.4),
                                nn.Linear(in_features=256, out_features=3, bias=False),
                                nn.LogSoftmax(dim=1))
        model = model.to(device) 
    elif model_name == 'vit_base':
        model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=False, num_classes=3)
        model.head = nn.Sequential(nn.Linear(in_features=768, out_features=256, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.4),
                                nn.Linear(in_features=256, out_features=3, bias=False),
                                nn.LogSoftmax(dim=1))
        model = model.to(device)
    elif model_name == 'eva02_base':
        model = timm.create_model("eva02_base_patch14_224.mim_in22k", pretrained=True)
        model = model.to(device)
    else:
        logger.error(f'{model_name} is invalid')
        sys.exit()
    
    if model_mode == "train":
        model = load_train_verify_model(config, model, device, logger)
        model = model.to(device)
    elif model_mode == "test":
        model = load_test_model(config, model, device, logger)
        model = model.to(device)
    else:
        logger.warning(f'the model mode {model_mode} is exist, can not load model weight')

    return model
