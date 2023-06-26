import json
import torch
from utils.dataset_util import verify_dataloader
from utils.device_util import devices
from utils.loss_util import loss_functions
from utils.logger_util import loggers
from utils.model_util import models
from utils.save_util import save_result_image_loss, create_eval_dir, save_loss_in_text
from utils.visualizer_util import Visualizers


def verify_gan(mg, vfld, eva, log, dev, cfg):
    mg.eval()
    total_loss = 0
    fn_list = []
    loss_list = []
    save_dir = create_eval_dir(cfg['verify_result_save_dir'])
    with torch.no_grad():
        for idx, (data, target, fn) in enumerate(vfld):
            target = target.to(dev)
            data = data.to(dev)
            output_g = mg(data)
            eval_loss = eva(output_g, target).data5d.item()
            total_loss += eval_loss
            fn_list.append(fn)
            loss_list.append(eval_loss)
            save_result_image_loss(save_dir, None, fn, output_g, eval_loss)
            Visual.vis_write('verify', {
                cfg['verify_loss_function']: eval_loss,
            }, idx)
    save_loss_in_text(save_dir, cfg['verify_save_filename'], fn_list, loss_list)
    avg_loss = total_loss / len(vfld.dataset)
    outputstring = 'Verify Average {}: {:.8f} '.format(type(eva).__name__, avg_loss)
    log.info(outputstring)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = loggers(config)

    Visual = Visualizers(config['visualization_verify'])

    device = devices(config, logger)

    verifyloader = verify_dataloader(config)

    generator_config = config['generator_model']
    g_model = models(generator_config, device, logger)

    evaluation = loss_functions(config['verify_loss_function'], logger)

    verify_gan(g_model, verifyloader, evaluation, logger, device, config)

    Visual.close_vis()
