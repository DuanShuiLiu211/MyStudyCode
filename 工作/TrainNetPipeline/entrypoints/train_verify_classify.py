import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import time
import torch
import json
from utils.dataset_util import train_verify_classify_dataloader
from utils.device_util import devices
from utils.logger_util import loggers
from utils.loss_util import loss_functions
from utils.model_util import get_model
from utils.optimizer_util import optimizers, adjust_learning_rate
from utils.save_util import save_tensor_to_image, save_model
from utils.visualizer_util import Visualizers


def train(md, epo, trld, opt, cri, vis, log, dev, cfg):
    md.train()
    total_loss = 0
    correct = 0
    idx = 0
    samples = 0
    for idx, (data, target, fn, raw) in enumerate(trld):
        inputs = data.to(dev)
        labels = target.to(dev)

        opt.zero_grad()
        outputs = md(inputs)
        loss = cri(outputs, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        samples += inputs.shape[0]

        outputstring = 'train epoch: {} and data batch: [{}~{}/{}], learn_rate: {:.8f}, {}: {:.8f}' \
            .format(epo, samples - inputs.shape[0] + 1, samples, len(trld.dataset),
                    opt.param_groups[0]['lr'], type(cri).__name__, loss.item())
        log.info(outputstring)

        if vis.savegraphs:
            vis.vis_graph(md, inputs)

        if (epo + 1) % int(cfg['epoch_print']) == 0:
            if (idx + 1) % int(cfg['batch_print']) == 0:
                if cfg['save_train_image']:
                    save_tensor_to_image(cfg['train_result_save_dir'], epo, fn, outputs, loss.item())
                    
        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

    avg_loss = total_loss / (idx + 1)
    avg_accuracy = correct / len(trld.dataset)

    vis.vis_write('train', {
        cfg['train_loss_function']: avg_loss,
        "train_accuracy": avg_accuracy,
    }, epo)

    outputstring = 'train epoch: {}, average {}: {:.8f} and average accuracy {:.8f}'.format(epo, type(cri).__name__, avg_loss, avg_accuracy)
    log.info(outputstring)

    with open(cfg['train_save_filename'], 'a') as fl:
        fl.write('{}:{} {}\n'.format(epo, avg_loss, avg_accuracy))

    return avg_loss, avg_accuracy


def verify(md, epo, tsld, eva, vis, log, dev, cfg):
    md.eval()
    total_loss = 0
    correct = 0
    idx = 0
    samples = 0
    with torch.no_grad():
        for idx, (data, target, fn, raw) in enumerate(tsld):
            inputs = data.to(dev)
            labels = target.to(dev)

            outputs = md(inputs)
            loss = eva(outputs, labels)
            
            total_loss += loss.item()
            samples += inputs.shape[0]

            outputstring = 'verify data batch: [{}~{}/{}], {}: {:.8f}' \
                .format(samples - inputs.shape[0] + 1, samples, len(tsld.dataset), type(eva).__name__, loss.item())
            log.info(outputstring)

            if (epo + 1) % int(cfg['epoch_print']) == 0:
                if (idx + 1) % int(cfg['batch_print']) == 0:
                    if cfg['save_verify_image']:
                        save_tensor_to_image(cfg['verify_result_save_dir'], epo, fn, outputs, loss.item())
            
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

    avg_loss = total_loss / (idx + 1)
    avg_accuracy = correct / len(tsld.dataset)

    outputstring = 'verify data average {}: {:.8f} and average accuracy {:.8f}'.format(type(eva).__name__, avg_loss, avg_accuracy)
    log.info(outputstring)

    vis.vis_write('verify', {
        cfg['verify_loss_function']: avg_loss,
        "verify_accuracy": avg_accuracy,
    }, epo)

    with open(cfg['verify_save_filename'], 'a') as fl:
        fl.write('{}:{} {}\n'.format(epo, avg_loss, avg_accuracy))

    return avg_loss, avg_accuracy


if __name__ == '__main__':
    """初始化环境与模型"""
    with open('../config.json', 'r') as f:
        config = json.load(f)
    
    logger = loggers(config)

    visual = Visualizers(config['visualization_train_verify'])

    device = devices(config, logger)

    trainloader, verifyloader = train_verify_classify_dataloader(config)

    criterion = loss_functions(config['train_loss_function'], logger)
    
    evaluation = loss_functions(config['verify_loss_function'], logger)
    
    general_config = config['general_model']
    model = get_model(general_config, device, logger)
    
    optimizer = optimizers(general_config, model, logger)

    scheduler = adjust_learning_rate(general_config, optimizer, logger)

    start_time = time.time()
    """模型周期迭代优化"""
    best_train_loss = float('Inf')
    best_train_average = float('-Inf')
    best_verify_loss = float('Inf')
    best_verify_average = float('-Inf')
    for epoch in range(int(config['epochs'])):
        # 训练模型
        train_avg_loss, train_avg_average = train(model, epoch, trainloader, optimizer, criterion, visual, logger, device, config)

        # 测试模型
        verify_avg_loss, verify_avg_average = verify(model, epoch, verifyloader, evaluation, visual, logger, device, config)

        # 保存模型
        if verify_avg_average > best_verify_average:
            best_verify_average = verify_avg_average
            save_model(general_config, model, optimizer, epoch, best_verify_average)

        # 调整学习率
        if general_config['scheduler_mode'] != 'auto':
            scheduler.step()
        elif general_config['scheduler_mode'] == 'auto':
            scheduler.step(metrics=verify_avg_loss)
        else:
            print("the lr scheduler mode is invalid")
            sys.exit()

    visual.close_vis()
    end_time = time.time()
    print(f'training total cost time: {(end_time - start_time):.2f} second')
