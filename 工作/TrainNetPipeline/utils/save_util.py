# encoding:utf-8
import os
import sys
import time
import torch
import numpy as np
from torchvision import utils as vutils
from scipy.io import savemat
from PIL import Image


def create_save_dir(save_dir):
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    if not os.path.exists(save_dir):
        save_dir = os.path.join(save_dir, f'{timestamp}_0')
        os.makedirs(save_dir)
    else:
        fn_list = list(map(int, os.listdir(save_dir)))
        if len(fn_list) == 0:
            save_dir = os.path.join(save_dir, f'{timestamp}_0')
        else:
            save_dir = os.path.join(save_dir, f'{timestamp}_{len(fn_list)+1}')
        os.makedirs(save_dir)
    return save_dir


def save_grid_image(tensor_img, img_path, norm=False, to_gray=False):
    if tensor_img.device != 'cpu':
        data = tensor_img.clone().detach().cpu()
    else:
        data = tensor_img.clone().detach()
        
    # make_grid 的输入 data 可以是 [b c h w]/[c h w]/[h w] 的形状
    grid = vutils.make_grid(data, nrow=8, padding=2, normalize=norm, range=None, scale_each=False, pad_value=0)
    if norm:
        ndarray = grid.mul(255).permute(1, 2, 0).to(dtype=torch.uint8).numpy()
    else:
        # 加 0.5 使其四舍五入到 [0,255] 中最接近的整数
        ndarray = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(dtype=torch.uint8).numpy()
        
    pil_img = Image.fromarray(ndarray)
    if to_gray:
        # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
        pil_img.convert('L').save(img_path)
    else:
        pil_img.save(img_path)


def save_tensor_to_image(base_path, tag, fn_list, tensor_img, aim, loss=None, format="rgb", save_raw=False):
    save_dir = os.path.join(base_path, f'{tag}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 先切断数据联系再切断计算图联系
    if tensor_img.device != 'cpu':
        data = tensor_img.clone().detach().cpu()
    else:
        data = tensor_img.clone().detach()

    if loss is not None:
        if loss.device != 'cpu':
            loss = round(loss.clone().detach().cpu().item(), 6)
        else:
            loss = round(loss.clone().detach().item(), 6)

    # 文件命名逻辑-[任务名称]_[平均损失]_[文件名]
    for idx, fn in enumerate(fn_list):
        filename, ext = os.path.splitext(fn)
        if ext == '.mat':
            if save_raw:
                savemat(os.path.join(save_dir, f'{aim}_{loss}_{fn}'), {f'{filename}': data[idx].numpy()})

        elif ext == '.npy':
            if save_raw:
                np.save(os.path.join(save_dir, f'{aim}_{loss}_{fn}'), data[idx].numpy())

        if len(data.shape) == 4:
            if format == "rgb":
                save_dir_sub = os.path.join(save_dir, f'{aim}_{loss}_{filename}')
                if not os.path.exists(save_dir_sub):
                    os.makedirs(save_dir_sub)
                for k, img in enumerate(data):
                    save_grid_image(img, os.path.join(save_dir_sub, f'{k}.png'))
            else:
                print("image format error")
                sys.exit()
        elif len(data.shape) == 3:
            if format == "rgb":
                save_grid_image(data, os.path.join(save_dir, f'{aim}_{loss}_{filename}.png'))
            elif format == "gray":
                save_dir_sub = os.path.join(save_dir, f'{aim}_{loss}_{filename}')
                if not os.path.exists(save_dir_sub):
                    os.makedirs(save_dir_sub)
                for k, img in enumerate(data):
                    save_grid_image(img, os.path.join(save_dir_sub, f'{k}.png'), norm=False, to_gray=True)
            else:
                print("image format error")
                sys.exit()
        elif len(data.shape) == 2:
            save_grid_image(data, os.path.join(save_dir, f'{aim}_{loss}_{filename}.png'), norm=False, to_gray=True)
        else:
            print("ndarray shape error")
            sys.exit()


def save_model(base_path, config, model, optimizer, epoch, loss=None):
    save_path = os.path.join(base_path, config['model_save_dir'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_mode = config['model_save_mode']
    aim = config['model_aim']
    mn = type(model).__name__
    opt = type(optimizer).__name__
    lr = optimizer.param_groups[0]['lr']
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    # 文件命名逻辑-[任务名称]_[模型名称]_[优化器名称]_[平均损失]_[周期]_[学习率]_[时间戳]
    if loss is not None:
        if save_mode == "model":
            torch.save(model, os.path.join(save_path, f'{aim}_{mn}_{opt}_{loss:.6f}_{epoch}_{lr}_{timestamp}.pth')) 
        elif save_mode == "weight":
            torch.save(model.state_dict(), os.path.join(save_path, f'{aim}_{mn}_{opt}_{loss:.6f}_{epoch}_{lr}_{timestamp}.pth'))
        elif save_mode == "checkpoint":
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch_current': epoch,}
            torch.save(checkpoint, os.path.join(save_path, f'{aim}_{mn}_{opt}_{loss:.6f}_{epoch}_{lr}_{timestamp}.pth'))
        else:
            pass
    # 文件命名逻辑-[任务名称]_[模型名称]_[优化器名称]_[周期]_[学习率]_[时间戳]
    else:
        if save_mode == "model":
            torch.save(model, os.path.join(save_path, f'{aim}_{mn}_{opt}_{epoch}_{lr}_{timestamp}.pth')) 
        elif save_mode == "weight":
            torch.save(model.state_dict(), os.path.join(save_path, f'{aim}_{mn}_{opt}_{epoch}_{lr}_{timestamp}.pth'))
        elif save_mode == "checkpoint":
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch_current': epoch,}
            torch.save(checkpoint, os.path.join(save_path, f'{aim}_{mn}_{opt}_{epoch}_{lr}_{timestamp}.pth'))
        else:
            pass