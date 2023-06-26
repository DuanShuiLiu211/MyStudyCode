import sys
import time
import torch
import json
sys.path.append(r"C:\Users\10469\Desktop\WorkFile\Code\TrainNetPipeline-main")
from utils.dataset_util import train_verify_classify_dataloader
from utils.device_util import devices
from utils.logger_util import loggers
from utils.loss_util import loss_functions
from utils.model_util import models
from utils.optimizer_util import optimizers, adjust_learning_rate
from utils.save_util import save_result_image_loss, save_model
from utils.visualizer_util import Visualizers


def train(md, epo, trld, opt, cri, vis, log, dev, cfg):
    md.train()
    total_loss = 0
    samples = 0
    idx = 0
    for idx, (data, target, fn) in enumerate(trld):
        inputs = data.to(dev)
        labels = target.to(dev)

        opt.zero_grad()
        outputs = md(inputs)
        loss = cri(outputs, labels)
        loss.backward()
        opt.step()

        total_loss += loss.data.item()
        samples += inputs.shape[0]

        outputstring = 'Train epoch: {} batch: [{}~{}/{}], learn_rate: {:.8f}, {}: {:.8f}' \
            .format(epo, samples - inputs.shape[0] + 1, samples, len(trld.dataset),
                    opt.param_groups[0]['lr'], type(cri).__name__, loss.data.item())
        log.info(outputstring)

        if vis.savegraphs:
            vis.vis_graph(md, inputs)

        if (epo + 1) % int(cfg['epoch_print']) == 0:
            if (idx + 1) % int(cfg['batch_print']) == 0:
                if cfg['save_train_image']:
                    save_result_image_loss(cfg['train_result_save_dir'], epo, fn, outputs, loss.data.item())

    avg_loss = total_loss / (idx + 1)

    vis.vis_write('train', {
        cfg['train_loss_function']: avg_loss,
    }, epo)

    outputstring = 'train epoch: {}, average {}: {:.8f}'.format(epo, type(cri).__name__, avg_loss)
    log.info(outputstring)

    with open('train_loss.txt', 'a') as fl:
        fl.write('{}:{}\n'.format(epo, avg_loss))

    return avg_loss


def verify(md, epo, tsld, eva, vis, log, dev, cfg):
    md.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target, fn) in enumerate(tsld):
            inputs = data.to(dev)
            labels = target.to(dev)

            outputs = md(inputs)
            loss = eva(outputs, labels)
            total_loss += loss.data.item()

            outputstring = 'verify data: {}, {}: {:.8f}'.format(idx + 1, type(eva).__name__, loss.data.item())
            log.info(outputstring)

            if (epo + 1) % int(cfg['epoch_print']) == 0:
                if (idx + 1) % int(cfg['batch_print']) == 0:
                    if cfg['save_verify_image']:
                        save_result_image_loss(cfg['verify_result_save_dir'], epo, fn, outputs, loss.data.item())
            
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

    avg_loss = total_loss / len(tsld.dataset)
    accuracy = correct / len(tsld.dataset)

    outputstring = 'verify data average {}: {:.8f} and accuracy {}'.format(type(eva).__name__, avg_loss, accuracy)
    log.info(outputstring)

    vis.vis_write('verify', {
        cfg['verify_loss_function']: avg_loss,
        "accuracy": accuracy,
    }, epo)

    with open('verify_loss.txt', 'a') as fl:
        fl.write('{}:{} {}\n'.format(epo, avg_loss, accuracy))

    return avg_loss


if __name__ == '__main__':
    """初始化环境与模型"""
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    logger = loggers(config)

    visual = Visualizers(config['visualization_train_verify'])

    device = devices(config, logger)

    trainloader, verifyloader = train_verify_classify_dataloader(config)

    general_config = config['general_model']
    model = models(general_config, device, logger)

    criterion = loss_functions(config['train_loss_function'], logger)
    best_train_loss = float('Inf')

    evaluation = loss_functions(config['verify_loss_function'], logger)
    best_verify_loss = float('Inf')

    optimizer = optimizers(general_config, model, logger)

    scheduler = adjust_learning_rate(general_config, optimizer, logger)

    start_time = time.time()
    """模型周期迭代优化"""
    for epoch in range(int(config['epochs'])):
        # 训练模型
        train_avg_loss = train(model, epoch, trainloader, optimizer, criterion, visual, logger, device, config)

        # 测试模型
        verify_avg_loss = verify(model, epoch, verifyloader, evaluation, visual, logger, device, config)

        # 保存模型
        if verify_avg_loss < best_verify_loss:
            best_verify_loss = verify_avg_loss
            save_model(config, model, optimizer, epoch, best_verify_loss)

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
