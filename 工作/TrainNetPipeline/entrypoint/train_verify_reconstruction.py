import sys
import time
import torch
import json
from utils.dataset_util import train_test_dataloader
from utils.device_util import devices
from utils.logger_util import loggers
from utils.loss_util import loss_functions
from utils.model_util import models
from utils.optimizer_util import optimizers, adjust_learning_rate
from utils.save_util import save_result_image_loss, save_model
from utils.visualizer_util import Visualizers


def train(md, epo, trld, opt, cri1, cri2, vis, log, dev, cfg):
    md.train()
    total_loss = 0
    samples = 0
    idx = 0
    for idx, (data, target, fn) in enumerate(trld):
        data = data.unsqueeze(1)

        data = data.to(dev)
        pseudo_psf = pseudo_psf.to(dev)
        label_sr = target[0].to(dev)
        label_ab = target[1].to(dev)

        opt.zero_grad()
        output_sr, output_ab = md(pseudo_psf)
        loss_sr = cri1(output_sr, label_sr)
        loss_ab = cri2(output_ab, label_ab)
        s_a = 1
        loss = loss_sr + s_a * loss_ab
        loss.backward()
        opt.step()

        total_loss += loss.data.item()
        samples += data.shape[0]

        outputstring = 'Train epoch: {} batch: [{}~{}/{}], learn_rate: {:.8f}, sr-{}: {:.8f}, ab-{}: {:.8f}' \
            .format(epo, samples - data.shape[0] + 1, samples, len(trld.dataset), opt.param_groups[0]['lr'],
                    type(cri1).__name__, loss_sr.data5d.item(), type(cri2).__name__, loss_ab.data5d.item())
        log.info(outputstring)

        if epo % int(cfg['epoch_print']) == 0:
            if idx % int(cfg['batch_print']) == 0:
                if cfg['save_train_image']:
                    save_result_image_loss(cfg['train_result_save_dir'], epo, fn,
                                           output_sr[:, 0, ...], loss_sr.data5d.item(), "sr")
                    save_result_image_loss(cfg['train_result_save_dir'], epo, fn,
                                           output_ab[:, 0, ...], loss_ab.data5d.item(), "ab")

    avg_loss = total_loss / (idx + 1)

    vis.vis_write('train', {
        cfg['train_loss_function_1']: avg_loss,
    }, epo)

    outputstring = 'Train epoch: {}, average {}+{}: {:.8f}'.format(epo, type(cri1).__name__,
                                                                   type(cri2).__name__, avg_loss)
    log.info(outputstring)

    with open('train_loss.txt', 'a') as fl:
        fl.write('{}:{}\n'.format(epo, avg_loss))

    return avg_loss


def test(md, epo, tsld, eva1, eva2, vis, log, dev, cfg):
    md.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (data, target, fn) in enumerate(tsld):
            data = data.unsqueeze(1)

            data = data.to(dev)
            pseudo_psf = pseudo_psf.to(dev)
            data = data.to(dev)
            label_sr = target[0].to(dev)
            label_ab = target[1].to(dev)

            output_sr, output_ab = md(data, pseudo_psf)
            loss_sr = eva1(output_sr[:, 0, ...], label_sr.repeat(1, 15, 1, 1))
            loss_ab = eva2(output_ab[:, 0, ...], label_ab[:, 0, ...])
            s_a = 1
            loss = loss_sr + s_a * loss_ab
            total_loss += loss.data.item()

            outputstring = 'Test data: {}, {}: {:.8f} {}: {:.8f}'.format(idx + 1,
                                                                         type(eva1).__name__, loss_sr.data5d.item(),
                                                                         type(eva2).__name__, loss_ab.data5d.item())
            log.info(outputstring)

            if epo % int(cfg['epoch_print']) == 0:
                if idx % int(cfg['batch_print']) == 0:
                    if cfg['save_test_image']:
                        save_result_image_loss(cfg['test_result_save_dir'], epo, fn,
                                               output_sr[:, 0, ...].detach(), loss_sr.data5d.item(), "sr")
                        save_result_image_loss(cfg['test_result_save_dir'], epo, fn,
                                               output_ab[:, 0, ...].detach(), loss_ab.data5d.item(), "ab")

    avg_loss = total_loss / len(tsld.dataset)

    outputstring = 'Test data average {}+{}: {:.8f} '.format(type(eva1).__name__, type(eva2).__name__, avg_loss)
    log.info(outputstring)

    vis.vis_write('test', {
        cfg['test_loss_function_1']: avg_loss,
    }, epo)

    with open('test_loss.txt', 'a') as fl:
        fl.write('{}:{}\n'.format(epo, avg_loss))

    return avg_loss


if __name__ == '__main__':
    """初始化环境与模型"""
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = loggers(config)

    visual = Visualizers(config['visualization_train_test'])

    device = devices(config, logger)

    trainloader, testloader = train_test_dataloader(config)

    general_config = config['general_model']
    model = models(general_config, device, logger)

    criterion_1 = loss_functions(config['train_loss_function'], logger)
    criterion_2 = loss_functions(config['train_loss_function'], logger)
    best_train_loss = float('Inf')

    evaluation_1 = loss_functions(config['test_loss_function'], logger)
    evaluation_2 = loss_functions(config['test_loss_function'], logger)
    best_test_loss = float('Inf')

    optimizer = optimizers(general_config, model, logger)

    scheduler = adjust_learning_rate(general_config, optimizer, logger)

    start_time = time.time()
    """模型周期迭代优化"""
    for epoch in range(int(config['epochs'])):
        # 训练模型
        train_avg_loss = train(model, epoch, trainloader, optimizer, criterion_1, criterion_2, visual, logger, device, config)

        # 测试模型
        test_avg_loss = test(model, epoch, testloader, evaluation_1, evaluation_2, visual, logger, device, config)

        # 保存模型
        if test_avg_loss < best_test_loss:
            best_test_loss = test_avg_loss
            save_model(config, model, optimizer, epoch, best_test_loss)

        # 调整学习率
        if general_config['scheduler_mode'] != 'auto':
            scheduler.step()
        elif general_config['scheduler_mode'] == 'auto':
            scheduler.step(metrics=test_avg_loss)
        else:
            print("The lr scheduler mode is invalid")
            sys.exit()

    visual.close_vis()
    end_time = time.time()
    print(f'Training total cost time: {(end_time - start_time):.2f} second')
