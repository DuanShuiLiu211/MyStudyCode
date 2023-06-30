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
from utils.model_util import get_model, load_train_verify_model
from utils.optimizer_util import optimizers, adjust_learning_rate
from utils.save_util import save_tensor_to_image, create_save_dir, save_model
from utils.visualizer_util import Visualizers


def train(md, epo, trld, opt, cri, tvsd, vis, log, dev, cfg):
    md.train()
    total_loss = 0
    correct = 0
    batch = 0
    samples = 0
    for batch, (data, target, fn, raw) in enumerate(trld):
        inputs = data.to(dev)
        labels = target.to(dev)

        opt.zero_grad()
        outputs = md(inputs)
        loss = cri(outputs, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
        samples += inputs.shape[0]

        outputstring = 'train data epoch: {}, batch: {}[{}~{}/{}], learn_rate: {:.8f}, {}: {:.8f}' \
            .format(epo, batch, samples - inputs.shape[0] + 1, samples, len(trld.dataset),
                    opt.param_groups[0]['lr'], type(cri).__name__, loss.item())
        log.info(outputstring)

        if vis.savegraphs:
            vis.vis_graph(md, inputs)

        if (epo + 1) % int(cfg['epoch_print_gap_train']) == 0:
            if (batch + 1) % int(cfg['batch_print_gap_train']) == 0:
                if cfg['save_image_train']:
                    save_tensor_to_image(tvsd, f'train_image_epoch_{epo}', fn, outputs, loss.item())

    avg_loss = total_loss / (batch + 1)
    avg_accuracy = correct / len(trld.dataset)

    vis.vis_write('train', {
        type(cri).__name__: avg_loss,
        "train_accuracy": avg_accuracy,
    }, epo)

    outputstring = 'train data epoch: {}, average {}: {:.8f}, average accuracy: {:.8f}' \
        .format(epo, type(cri).__name__, avg_loss, avg_accuracy)
    log.info(outputstring)

    with open(os.path.join(tvsd, cfg['loss_filename_train_verify']), 'a') as fl:
        fl.write(f'train data epoch: {epo}\n{type(cri).__name__}: {avg_loss}, accuracy: {avg_accuracy}\n')

    return avg_loss, avg_accuracy


def verify(md, epo, tsld, eva, tvsd, vis, log, dev, cfg):
    md.eval()
    total_loss = 0
    correct = 0
    batch = 0
    samples = 0
    with torch.no_grad():
        for batch, (data, target, fn, raw) in enumerate(tsld):
            inputs = data.to(dev)
            labels = target.to(dev)

            outputs = md(inputs)
            loss = eva(outputs, labels)
            
            total_loss += loss.item()
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            samples += inputs.shape[0]

            outputstring = 'verify data epoch: {}, batch: {}[{}~{}/{}], {}: {:.8f}' \
                .format(epo, batch, samples - inputs.shape[0] + 1, samples, len(tsld.dataset), type(eva).__name__, loss.item())
            log.info(outputstring)

            if (epo + 1) % int(cfg['epoch_print_gap_verify']) == 0:
                if (batch + 1) % int(cfg['batch_print_gap_verify']) == 0:
                    if cfg['save_image_verify']:
                        save_tensor_to_image(tvsd, f'verify_image_epoch_{epo}', fn, outputs, loss.item())

    avg_loss = total_loss / (batch + 1)
    avg_accuracy = correct / len(tsld.dataset)

    outputstring = 'verify data epoch: {}, average {}: {:.8f}, average accuracy: {:.8f}' \
        .format(epo, type(eva).__name__, avg_loss, avg_accuracy)
    log.info(outputstring)

    vis.vis_write('verify', {
        type(eva).__name__: avg_loss,
        "verify_accuracy": avg_accuracy,
    }, epo)

    with open(os.path.join(tvsd, cfg['loss_filename_train_verify']), 'a') as fl:
        fl.write(f'verify data epoch: {epo}\n{type(eva).__name__}: {avg_loss}, accuracy: {avg_accuracy}\n')

    return avg_loss, avg_accuracy


if __name__ == '__main__':
    """初始化环境与模型"""
    with open('../config.json', 'r') as f:
        config = json.load(f)
    train_verify_save_dir = create_save_dir(config['train_verify_result_save_dir'])
    logger = loggers(train_verify_save_dir, config['log_filename_train_verify'], config)
    visual = Visualizers(train_verify_save_dir, config['visualization_filename_train_verify'])
    device = devices(config, logger)
    trainloader, verifyloader = train_verify_classify_dataloader(config)

    general_config = config['general_model']
    model = get_model(general_config, device, logger)
    optimizer = optimizers(general_config, model, logger)
    epoch_current = int(general_config['epoch_current'])
    epoch_end = int(general_config['epoch_end'])
    model, optimizer, epoch_current = load_train_verify_model(general_config, model, optimizer, epoch_current, device, logger)
    scheduler = adjust_learning_rate(general_config, optimizer, logger)
    criterion = loss_functions(general_config['train_loss_function'], logger)
    evaluation = loss_functions(general_config['verify_loss_function'], logger)
    
    """模型周期迭代优化"""
    start_time = time.time()
    best_train_loss = float('Inf')
    best_train_accuracy = float('-Inf')
    best_verify_loss = float('Inf')
    best_verify_accuracy = float('-Inf')
    for epoch in range(epoch_current, epoch_end):
        # 训练模型
        train_avg_loss, train_avg_average = train(model, epoch, trainloader, optimizer, criterion, train_verify_save_dir, visual, logger, device, config)

        # 测试模型
        verify_avg_loss, verify_avg_average = verify(model, epoch, verifyloader, evaluation, train_verify_save_dir, visual, logger, device, config)

        # 保存模型
        if verify_avg_average > best_verify_accuracy:
            best_verify_accuracy = verify_avg_average
            save_model(train_verify_save_dir, general_config, model, optimizer, epoch, best_verify_accuracy)

        # 调整学习率
        if general_config['scheduler_mode'] != 'auto':
            scheduler.step()
        elif general_config['scheduler_mode'] == 'auto':
            scheduler.step(metrics=verify_avg_loss)
        else:
            print("the lr scheduler mode is invalid")
            sys.exit()

    end_time = time.time()
    print(f'training total cost time: {(end_time - start_time):.2f} second')
    visual.close_vis()
