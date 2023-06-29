#
import os
import numpy as np
import PIL.Image as Image

def save_result_image(save_dir, epoch, filenames, images):
    save_dir = os.path.join(save_dir, f'epoch_{epoch}')

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    for idx, fn in enumerate(filenames):
        img = (images[idx, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_dir, fn))

def save_result_image_with_metric(save_dir, fnList, images, metric):
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    for idx, fn in enumerate(fnList):
        img = (images[idx, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_dir, f'{metric:.6f}_{fn}'))

def create_eval_dir(save_dir):
    if os.path.exists(save_dir) == False:
        save_dir = os.path.join(save_dir,'0')
        os.makedirs(save_dir)
    else:
        fnList = list(map(int,os.listdir(save_dir)))
        if len(fnList) == 0:
            save_dir = os.path.join(save_dir, '0')
        else:
            save_dir = os.path.join(save_dir, str(max(fnList)+1))
        os.makedirs(save_dir)

    return save_dir

def save_results_in_file(save_dir,save_filename,fnList,metricList):
    save_strings = ''
    for fn,metric in zip(fnList,metricList):
        save_strings += f'{fn[0]} {metric}\n'

    with open(os.path.join(save_dir,save_filename),'w') as f:
        f.write(save_strings)