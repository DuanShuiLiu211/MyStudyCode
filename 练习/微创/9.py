import numpy as np
import os
import cv2
from skimage import morphology


def file_remove(file_list):
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        pass
    try:
        file_list.remove('._.DS_Store')
    except ValueError:
        pass


def remove_isolate(inputs, threshold_area=0.5):
    """
    inputs: x*y*3
    """
    mask = np.zeros((inputs.shape[0], inputs.shape[1]), dtype=np.uint8)
    mask[np.sum(inputs, axis=-1) > 0] = 1
    mask = morphology.remove_small_objects(mask.astype(np.bool8),
                                           np.sum(mask) * threshold_area,
                                           connectivity=8).astype(np.uint8)
    outputs = np.expand_dims(mask, -1) * inputs

    return outputs


def crop_array(image, label, up, down, left, right):
    """
    image:x*y*3 uint8
    label:x*y*3 uint8
    """

    crop_image = image[up:down, left:right, :]
    crop_label = label[up:down, left:right, :]

    return crop_image, crop_label


def count_list_place(base_path, files):
    place = []
    file_dir = f"{base_path}/{files}"
    path_4 = os.listdir(file_dir)
    file_remove(path_4)
    for file_name in path_4:
        if "._" not in file_name:
            data = cv2.imdecode(
                np.fromfile(f"{file_dir}/{file_name}", dtype=np.uint8), -1)

            data = remove_isolate(data)
            data_gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

            y_sum = np.sum(data_gray, axis=1)
            x_sum = np.sum(data_gray, axis=0)

            up = min(np.where(y_sum != 0)[0]) - 20 if min(np.where(y_sum != 0)[0]) > 20 else 0
            down = max(np.where(y_sum != 0)[0]) + 20 if max(np.where(y_sum != 0)[0]) < data.shape[0] - 20 else data.shape[0]
            left = min(np.where(x_sum != 0)[0]) - 20 if min(np.where(x_sum != 0)[0]) > 20 else 0
            right = max(np.where(x_sum != 0)[0]) + 20 if min(np.where(x_sum != 0)[0]) < data.shape[1] - 20 else data.shape[1]

        place.append([up, down, left, right])

    place = np.array(place)
    place_min = np.min(place, axis=0)
    place_max = np.max(place, axis=0)
    up, down, left, right = place_min[0], place_max[1], place_min[2], place_max[3]
    assert (down < data.shape[0]) and (right < data.shape[1])

    return up, down, left, right


def count_place(data, path):
    data_gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

    y_sum = np.sum(data_gray, axis=1)
    x_sum = np.sum(data_gray, axis=0)
    try:
        up = min(np.where(y_sum != 0)[0]) - 20 if min(np.where(y_sum != 0)[0]) > 20 else 0
        down = max(np.where(y_sum != 0)[0]) + 20 if max(np.where(y_sum != 0)[0]) < data.shape[0] - 20 else data.shape[0]
        left = min(np.where(x_sum != 0)[0]) - 20 if min(np.where(x_sum != 0)[0]) > 20 else 0
        right = max(np.where(x_sum != 0)[0]) + 20 if min(np.where(x_sum != 0)[0]) < data.shape[1] - 20 else data.shape[1]
        flag = 0
    except ValueError:
        flag = 1
        print(path)

    return up, down, left, right, flag


file_path = '/Volumes/昊大侠/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/短轴动态狭窄率/dataset_0802'
path_1 = os.listdir(file_path)
file_remove(path_1)
for dir_path in path_1:
    if dir_path == 'label':
        label_dir_path = os.path.join(file_path, dir_path)
        path_2 = os.listdir(label_dir_path)
        file_remove(path_2)
        for label_path in path_2:
            if "._" not in label_path:
                # up, down, left, right = count_list_place(label_dir_path, label_path)

                label_dir = f"{label_dir_path}/{label_path}"
                image_dir = label_dir.replace("label", "image")

                save_label_path = label_dir.replace('dataset_0802', 'dataset_roi_0802')
                if not os.path.exists(save_label_path):
                    os.mkdir(save_label_path)

                save_image_path = image_dir.replace('dataset_0802', 'dataset_roi_0802')
                if not os.path.exists(save_image_path):
                    os.mkdir(save_image_path)

                path_3 = os.listdir(label_dir)
                file_remove(path_3)
                for file_name in path_3:
                    if "._" not in file_name:
                        label = cv2.imdecode(
                            np.fromfile(f"{label_dir}/{file_name}", dtype=np.uint8), -1)
                        label = remove_isolate(label)
                        image = cv2.imdecode(
                            np.fromfile(f"{image_dir}/{file_name}", dtype=np.uint8), -1)

                        up, down, left, right, flag = count_place(label, image_dir)

                        if flag:
                            break

                        crop_image, crop_label = crop_array(image, label, up, down, left, right)

                        cv2.imencode('.png', crop_image)[1].tofile(
                            os.path.join(save_image_path, file_name))
                        cv2.imencode('.png', crop_label)[1].tofile(
                            os.path.join(save_label_path, file_name))

print("运行结束")
