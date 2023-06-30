# encoding:utf-8
import os
import time
import copy
import gc
from torch.utils.tensorboard import SummaryWriter


class Visualizers:
    def __init__(self, visual_root, visual_dir, savegraphs=True):
        self.writer = SummaryWriter(os.path.join(visual_root, visual_dir))
        self.sub_root = time.strftime('%Y年%m月%d日%H时%M分%S秒', time.localtime())
        self.savegraphs = savegraphs

    def vis_write(self, main_tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(self.sub_root + '_{}'.format(main_tag), tag_scalar_dict, global_step)

    def vis_graph(self, model, input_to_model=None):
        model_copy = copy.deepcopy(model)
        if self.savegraphs:
            with self.writer as w:
                w.add_graph(model_copy, input_to_model)
                self.savegraphs = False
        del model_copy
        gc.collect()

    def vis_image(self, tag, img_tensor, epoch=None, step=None, formats='CHW'):
        if img_tensor.device != 'cpu':
            data = img_tensor.clone().detach().cpu()
        else:
            data = img_tensor.clone().detach()
        if epoch is not None:
            self.writer.add_image(self.sub_root + f'_{tag}_{epoch}', data, global_step=step, dataformats=formats)
        else:
            self.writer.add_image(self.sub_root + f'_{tag}', data, global_step=step, dataformats=formats)

    def vis_images(self, tag, img_tensor, epoch=None, step=None, formats='NCHW'):
        if img_tensor.device != 'cpu':
            data = img_tensor.clone().detach().cpu()
        else:
            data = img_tensor.clone().detach()
        if epoch is not None:
            self.writer.add_images(self.sub_root + f'_{tag}_{epoch}', data, global_step=step, dataformats=formats)
        else:
            self.writer.add_images(self.sub_root + f'_{tag}', data, global_step=step, dataformats=formats)

    def close_vis(self):
        self.writer.close()
