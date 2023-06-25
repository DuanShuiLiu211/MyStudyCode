import sys

import torch
import json
import numpy as np
from utils.dataset_util import train_test_dataloader
from utils.device_util import devices
from utils.logger_util import loggers
from utils.loss_util import loss_functions
from utils.model_util import models
from utils.optimizer_util import optimizers, adjust_learning_rate
from utils.save_util import save_result_image_loss, save_model
from utils.visualizer_util import Visualizers


def train_gan(mg, md, e, trld, optg, optd, crid, crig, log, dev, cfg, cfgg, cfgd):
    total_loss_g1 = 0
    total_loss_g2 = 0
    total_loss_d1 = 0
    total_loss_d2 = 0
    total_loss_d = 0
    samples = 0
    iter_nums = 1
    iter_g = cfgg['batch_iter_generator']
    iter_d = cfgd['batch_iter_discriminator']
    for idx, (data, target, fn) in enumerate(trld):
        data = data.to(dev)
        target = target.to(dev)
        target_d_t = torch.ones((target.shape[0], target.shape[1])).to(dev)
        target_d_f = torch.zeros((target.shape[0], target.shape[1])).to(dev)
        samples += data.shape[0]

        if idx == iter_g + iter_d:
            iter_g = iter_g + iter_d + cfgg['batch_iter_generator']
            iter_nums = iter_nums + 1

        if idx < iter_g:
            # stage_two：训练生成器
            mg.train()
            md.eval()
            optd.zero_grad()
            optg.zero_grad()

            g_data_f = mg(data)[:, 0, :, :].unsqueeze(1)
            d_data_f = md(g_data_f)

            # 描述生成的结果与真实的结果的相似程度
            loss_g1 = crig(g_data_f.detach(), target)
            total_loss_g1 += loss_g1.data5d.item()
            # 判别器损失变小则说明判别器认为生成的结果与真实的结果相似即生成器变强
            loss_g2 = crid(d_data_f, target_d_t)
            total_loss_g2 += loss_g2.data5d.item()
            loss_g2.backward()

            optg.step()

            outputstring = 'Train Epoch: {} Batch: [{}~{}/{}], Generator_learn_rate: {:.8f}, Generator {}_g1: {:.8f} {}_g2: {:.8f}' \
                .format(e, samples - data.shape[0] + 1, samples, len(trld.dataset), optg.param_groups[0]['lr'],
                        type(crig).__name__, loss_g1.data5d.item(), type(crid).__name__, loss_g2.data5d.item())
            log.info(outputstring)

            if (e + 1) % int(cfg['epoch_print']) == 0:
                if (idx + 1) % int(cfg['batch_print']) == 0:
                    if cfg['save_train_image']:
                        save_result_image_loss(cfg['train_result_save_dir_generator'], e, fn,
                                               g_data_f.detach(), loss_g1.data5d.item())

        elif idx < iter_g + iter_d:
            # stage_one：训练判别器
            mg.eval()
            md.train()
            optg.zero_grad()
            optd.zero_grad()

            g_data_f = mg(data)[:, 0, :, :].unsqueeze(1)
            d_data_f = md(g_data_f.detach())
            d_data_t = md(target)

            # 判别器损失变小则判别器学习了生成的结果是假的而真实的结果才是真的即判别器变强
            loss_d1 = crid(d_data_f, target_d_f)
            loss_d2 = crid(d_data_t, target_d_t)
            loss_d = (loss_d1 + loss_d2) / 2
            total_loss_d1 += loss_d1.data5d.item()
            total_loss_d2 += loss_d2.data5d.item()
            total_loss_d += loss_d.data.item()
            loss_d.backward()

            optd.step()

            outputstring = 'Train Epoch: {} Batch: [{}~{}/{}], Discriminator_learn_rate: {:.8f},' \
                           ' Discriminator {}_d1: {:.8f} {}_d2: {:.8f} {}_d: {:.8f}' \
                .format(e, samples - data.shape[0], samples, len(trld.dataset), optd.param_groups[0]['lr'],
                        type(crid).__name__, loss_d1.data5d.item(),
                        type(crid).__name__, loss_d2.data5d.item(),
                        type(crid).__name__, loss_d.data.item())
            log.info(outputstring)

        else:
            print("当前数据批次未进入训练策略")
            sys.exit()

    tail = len(trld.dataset) % (cfg['train_loader_batch'] * (cfgg['batch_iter_generator'] + cfgd['batch_iter_discriminator']))
    if tail != 0:
        if tail <= (cfgg['input_batch_size'] * cfgg['batch_iter_generator']):
            avg_loss_g1 = total_loss_g1 / ((iter_nums - 1) * cfgg['batch_iter_generator'] + np.ceil(tail / cfgg['input_batch_size']))
            avg_loss_g2 = total_loss_g2 / ((iter_nums - 1) * cfgg['batch_iter_generator'] + np.ceil(tail / cfgg['input_batch_size']))
            avg_loss_d1 = total_loss_d1 / ((iter_nums - 1) * (cfgd['batch_iter_discriminator'] + 1e-8))
            avg_loss_d2 = total_loss_d2 / ((iter_nums - 1) * (cfgd['batch_iter_discriminator'] + 1e-8))
            avg_loss_d = total_loss_d / ((iter_nums - 1) * (cfgd['batch_iter_discriminator'] + 1e-8))
        else:
            avg_loss_g1 = total_loss_g1 / (iter_nums * cfgg['batch_iter_generator'])
            avg_loss_g2 = total_loss_g2 / (iter_nums * cfgg['batch_iter_generator'])
            avg_loss_d1 = total_loss_d1 / (
                          (iter_nums - 1) * cfgd['batch_iter_discriminator'] +
                          np.ceil((tail - cfgg['input_batch_size'] * cfgg['batch_iter_generator']) / cfgd['input_batch_size']))
            avg_loss_d2 = total_loss_d2 / (
                    (iter_nums - 1) * cfgd['batch_iter_discriminator'] +
                    np.ceil((tail - cfgg['input_batch_size'] * cfgg['batch_iter_generator']) / cfgd['input_batch_size']))
            avg_loss_d = total_loss_d / (
                    (iter_nums - 1) * cfgd['batch_iter_discriminator'] +
                    np.ceil((tail - cfgg['input_batch_size'] * cfgg['batch_iter_generator']) / cfgd['input_batch_size']))
    else:
        avg_loss_g1 = np.ceil(total_loss_g1 / (iter_nums * (cfgg['batch_iter_generator'] + 1e-8)))
        avg_loss_g2 = np.ceil(total_loss_g2 / (iter_nums * (cfgg['batch_iter_generator'] + 1e-8)))
        avg_loss_d1 = np.ceil(total_loss_d1 / (iter_nums * (cfgd['batch_iter_discriminator'] + 1e-8)))
        avg_loss_d2 = np.ceil(total_loss_d2 / (iter_nums * (cfgd['batch_iter_discriminator'] + 1e-8)))
        avg_loss_d = np.ceil(total_loss_d / (iter_nums * (cfgd['batch_iter_discriminator'] + 1e-8)))

    visual.vis_write('train', {
        cfg['train_loss_function_generator']: avg_loss_g1,
        cfg['train_loss_function_discriminator']: avg_loss_g2,
        cfg['train_loss_function_discriminator']: avg_loss_d1,
        cfg['train_loss_function_discriminator']: avg_loss_d2,
        cfg['train_loss_function_discriminator']: avg_loss_d,
    }, e)

    outputstring = 'Train Epoch: {}, The Generator Average {}_g1: {:.8f} {}_g2: {:.8f},' \
                   ' The Discriminator Average {}_d1: {:.8f} {}_d2: {:.8f} {}_d: {:.8f}' \
        .format(e, type(crig).__name__, avg_loss_g1, type(crid).__name__, avg_loss_g2,
                type(crid).__name__, avg_loss_d1, type(crid).__name__, avg_loss_d2, type(crid).__name__, avg_loss_d)
    log.info(outputstring)

    with open('train_loss.txt', 'a') as fl:
        fl.write('{}:{}:{}:{}:{}\n'.format(e, avg_loss_g1, avg_loss_g2, avg_loss_d1, avg_loss_d2, avg_loss_d))


def test_gan(mg, md, e, tsld, evag, evad, log, dev, cfg):
    mg.eval()
    md.eval()
    total_loss_g1 = 0
    total_loss_d1 = 0
    total_loss_d2 = 0
    total_loss_d = 0
    with torch.no_grad():
        for idx, (data, target, fn) in enumerate(tsld):
            data = data.to(dev)
            target = target.to(dev)
            target_d_t = torch.ones((target.shape[0], target.shape[1])).to(dev)
            target_d_f = torch.zeros((target.shape[0], target.shape[1])).to(dev)
            g_data_f = mg(data)[:, 0, :, :].unsqueeze(1)
            d_data_f = md(g_data_f)
            d_data_t = md(target)

            loss_g1 = evag(g_data_f, target).data5d.item()
            loss_d1 = evad(d_data_f, target_d_f).data5d.item()
            loss_d2 = evad(d_data_t, target_d_t).data5d.item()
            loss_d = (loss_d1 + loss_d2) / 2

            outputstring = 'Test Data: {}, Generator {}_g1: {:.8f}, Discriminator {}_d1: {:.8f} {}_d2: {:.8f} {}_d: {:.8f}'\
                .format(idx + 1, type(evag).__name__, loss_g1, type(evag).__name__, loss_d1,
                        type(evag).__name__, loss_d2, type(evad).__name__, loss_d)
            log.info(outputstring)

            total_loss_g1 += loss_g1
            total_loss_d1 += loss_d1
            total_loss_d2 += loss_d2
            total_loss_d += loss_d

            if (e + 1) % int(cfg['epoch_print']) == 0:
                if cfg['save_test_image']:
                    save_result_image_loss(cfg['test_result_save_dir_generator'], e, fn, g_data_f, loss_g1)

    avg_loss_g1 = total_loss_g1 / (idx + 1)
    avg_loss_d1 = total_loss_d1 / (idx + 1)
    avg_loss_d2 = total_loss_d2 / (idx + 1)
    avg_loss_d = total_loss_d / (idx + 1)

    outputstring = 'Test Average Generator {}_g1: {:.8f}, Test Discriminator Average {}: {:.8f}  {}: {:.8f}  {}: {:.8f}'\
        .format(type(evag).__name__, avg_loss_g1,
                type(evad).__name__, avg_loss_d1,
                type(evad).__name__, avg_loss_d2,
                type(evad).__name__, avg_loss_d)
    log.info(outputstring)

    save_model(cfg, mg, e, avg_loss_g1)
    save_model(cfg, md, e, avg_loss_d)

    visual.vis_write('test_generator', {
        cfg['test_loss_function_generator']: avg_loss_g1,
        cfg['test_loss_function_discriminator']: avg_loss_d,
    }, e)

    with open('test_loss.txt', 'a') as fl:
        fl.write('{}:{}:{}:{}:{}\n'.format(e, avg_loss_g1, avg_loss_d1, avg_loss_d2, avg_loss_d))


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = loggers(config)

    visual = Visualizers(config['visualization_train_test'])

    device = devices(config, logger)

    trainloader, testloader = train_test_dataloader(config)

    generator_config = config['generator_model']
    discriminator_config = config['discriminator_model']
    g_model = models(generator_config, device, logger)
    d_model = models(discriminator_config, device, logger)

    criterion_g = loss_functions(config['train_loss_function_generator'], logger)
    criterion_d = loss_functions(config['train_loss_function_discriminator'], logger)

    evaluation_g = loss_functions(config['test_loss_function_generator'], logger)
    evaluation_d = loss_functions(config['test_loss_function_discriminator'], logger)

    g_optimizer = optimizers(generator_config, g_model, logger)
    d_optimizer = optimizers(discriminator_config, d_model, logger)

    scheduler_g = adjust_learning_rate(generator_config, g_optimizer, logger)
    scheduler_d = adjust_learning_rate(discriminator_config, d_optimizer, logger)

    for epoch in range(int(config['epochs'])):
        train_gan(g_model, d_model, epoch, trainloader, g_optimizer, d_optimizer, criterion_g, criterion_d, logger, device, config,
                  generator_config, discriminator_config)
        test_gan(g_model, d_model, epoch, testloader, evaluation_g, evaluation_d, logger, device, config)
        scheduler_g.step()
        scheduler_d.step()
    visual.close_vis()
