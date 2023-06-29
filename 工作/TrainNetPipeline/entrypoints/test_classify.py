import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import json
import torch
from utils.dataset_util import test_classify_dataloader
from utils.device_util import devices
from utils.loss_util import loss_functions
from utils.logger_util import loggers
from utils.model_util import get_model
from utils.save_util import save_tensor_to_image, create_save_dir, save_loss_in_text
from utils.visualizer_util import Visualizers


def test(m, vfld, eva, log, dev, cfg):
    m.eval()
    total_loss = 0
    correct = 0
    idx = 0
    samples = 0
    fn_list = []
    loss_list = []
    save_dir = create_save_dir(cfg['test_result_save_dir'])
    with torch.no_grad():
        for idx, (data, target, fn, raw) in enumerate(vfld):
            inputs = data.to(dev)
            labels = target.to(dev)
            raws = raw.squeeze(0).clone().detach()

            outputs = m(inputs)
            loss = eva(outputs, labels)

            total_loss += loss.item()
            samples += inputs.shape[0]

            fn_list.append(fn)
            loss_list.append(loss.item())

            outputstring = 'test data batch: [{}~{}/{}], {}: {:.8f}' \
                .format(samples - inputs.shape[0] + 1, samples, len(vfld.dataset), type(eva).__name__, loss.item())
            log.info(outputstring)

            if cfg['save_test_image']:
                save_tensor_to_image(save_dir, f'{torch.argmax(outputs, dim=1).item()}', fn, raws, cfg['model_aim'], loss)

            Visual.vis_write('test', {
                cfg['test_loss_function']: loss.item(),
            }, idx)

            Visual.vis_image(f'{torch.argmax(outputs, dim=1)}', raws, step=idx, formats='CHW')

            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

    save_loss_in_text(save_dir, cfg['test_save_filename'], fn_list, loss_list)

    avg_loss = total_loss / (idx + 1)
    avg_accuracy = correct / len(vfld.dataset)

    outputstring = 'test average {}: {:.8f} and average accuracy {:.8f}'.format(type(eva).__name__, avg_loss, avg_accuracy)
    log.info(outputstring)


if __name__ == '__main__':
    with open('../config.json', 'r') as f:
        config = json.load(f)

    logger = loggers(config)

    Visual = Visualizers(config['visualization_test'])

    device = devices(config, logger)

    verifyloader = test_classify_dataloader(config)

    general_config = config['general_model']
    
    model = get_model(general_config, device, logger)

    evaluation = loss_functions(config['test_loss_function'], logger)

    test(model, verifyloader, evaluation, logger, device, config)

    Visual.close_vis()