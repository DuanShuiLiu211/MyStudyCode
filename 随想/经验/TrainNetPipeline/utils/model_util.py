# encoding:utf-8
import torch
import sys
import time
import timm
from thop import profile


def load_model_weight(config, model, logger):
    logger.info(model)
    model_weight = config['last_model_weight']
    logger.info('Loading last weight from {} for model {}'.format(model_weight, type(model).__name__))
    try:
        new_weight_dict = torch.load(model_weight, map_location=torch.device('cpu'))
        old_weight_dict = model.state_dict()
        updated_weight_dict = {k: v for k, v in new_weight_dict.items() if k in old_weight_dict}
        old_weight_dict.update(updated_weight_dict)
        model.load_state_dict(old_weight_dict)

        new_params = len(new_weight_dict)
        old_params = len(old_weight_dict)
        matched_params = len(updated_weight_dict)
        logger.info('The new model params:{}, old model params:{}, matched params:{}'
                    .format(new_params, old_params, matched_params))
    except FileNotFoundError:
        logger.warning('Can not load last weight from {} for model {}'
                       .format(model_weight, type(model).__name__))
        logger.info('The parameters of model is initialized by method in model set')

    return model


def complexitys(config, model, logger):
    input_size = [config['input_batch_size'], *config["input_size"]]
    inputs = torch.randn(input_size)
    time_start = time.time_ns()
    try:
        macs, params = profile(model, inputs=(inputs, inputs))
    except TypeError:
        macs, params = profile(model, inputs=inputs)
    time_end = time.time_ns()
    take_time = time_end - time_start
    logger.info('MACs = ' + str(macs / 1e9) + 'G')
    logger.info('Times = ' + str(take_time / 1e6) + 'ms')

    blank = ' '
    logger.info('-' * 119)
    logger.info('|' + ' ' * 30 + 'weight name' + ' ' * 30 + '|' + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' + ' ' * 3 +
                'number' + ' ' * 3 + '|')
    logger.info('-' * 119)
    num_para = 0

    for index, (key, w_variable) in enumerate(model.named_parameters()):
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

    mn = type(model).__name__
    logger.info('-' * 119)
    logger.info('The parameters calculate of profile = ' + str(params/1e6) + 'M')
    logger.info(f'The parameters calculate named_parameters of {mn}: {num_para/1e6:.2f}M')
    logger.info('-' * 119)


def models(config, device, logger):
    if config['model_name'] == 'resnet18':
        model = timm.create_model("resnet18", pretrained=True).to(device)
    else:
        logger.error('{} is invalid'.format(config['model_name']))
        sys.exit()

    model = load_model_weight(config, model, logger)
    complexitys(config, model, logger)

    return model
