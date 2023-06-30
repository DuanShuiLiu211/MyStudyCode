import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import time
import json
import torch
from utils.dataset_util import test_classify_dataloader
from utils.device_util import devices
from utils.loss_util import loss_functions
from utils.logger_util import loggers
from utils.model_util import get_model, load_test_model
from utils.save_util import save_tensor_to_image, create_save_dir
from utils.visualizer_util import Visualizers


def test(m, vfld, eva, tsd, vis, log, dev, cfg):
    m.eval()
    total_loss = 0
    correct = 0
    batch = 0
    samples = 0
    fn_list = []
    loss_list = []
    with torch.no_grad():
        for batch, (data, target, fn, raw) in enumerate(vfld):
            inputs = data.to(dev)
            labels = target.to(dev)
            raws = raw.squeeze(0)
            
            outputs = m(inputs)
            loss = eva(outputs, labels)

            fn_list.append(fn)
            loss_list.append(loss.item())
            total_loss += loss.item()       
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            samples += inputs.shape[0]

            outputstring = 'test data batch: {}[{}~{}/{}], {}: {:.8f}' \
                .format(batch, samples - inputs.shape[0] + 1, samples, len(vfld.dataset), type(eva).__name__, loss.item())
            log.info(outputstring)

            if inputs.shape[0] > 1:
                vis.vis_images(f'{torch.argmax(outputs, dim=1)}', raws, step=batch, formats='NCHW')
            else:
                vis.vis_image(f'{torch.argmax(outputs, dim=1)}', raws, step=batch, formats='CHW')

            vis.vis_write('test', {
                type(eva).__name__: loss.item(),
            }, batch)

            if (batch + 1) % int(cfg['batch_print_gap_test']) == 0:
                if cfg['save_image_test']:
                    save_tensor_to_image(tsd, f'test_image_{torch.argmax(outputs, dim=1)}', fn, raws, cfg['model_aim'], loss)

        for fn, loss in zip(fn_list, loss_list):
            fn_file = ''
            for idx in range(len(fn)):
                fn_file += f'{fn[idx]} '
            with open(os.path.join(tsd, cfg['loss_filename_test']), 'a') as fl:
                fl.write(f'test data: {fn_file}\n{type(eva).__name__}: {loss}\n')
                
    avg_loss = total_loss / (batch + 1)
    avg_accuracy = correct / len(vfld.dataset)
    
    with open(os.path.join(tsd, cfg['loss_filename_test']), 'a') as fl:
         fl.write(f'test data average {type(eva).__name__}: {avg_loss}, average accuracy: {avg_accuracy}\n')

    outputstring = 'test data average {}: {:.8f}, average accuracy: {:.8f}' \
        .format(type(eva).__name__, avg_loss, avg_accuracy)
    log.info(outputstring)


if __name__ == '__main__':
    with open('../config.json', 'r') as f:
        config = json.load(f)
    test_save_dir = create_save_dir(config['test_result_save_dir'])
    logger = loggers(test_save_dir, config['log_filename_test'], config)
    visual = Visualizers(test_save_dir, config['visualization_filename_test'])
    device = devices(config, logger)
    verifyloader = test_classify_dataloader(config)
    
    general_config = config['general_model']
    model = get_model(general_config, device, logger)
    model = load_test_model(general_config, model, device, logger)
    evaluation = loss_functions(general_config['test_loss_function'], logger)

    start_time = time.time()
    test(model, verifyloader, evaluation, test_save_dir, visual, logger, device, config)
    end_time = time.time()
    print(f'training total cost time: {(end_time - start_time):.2f} second')
    visual.close_vis()
    