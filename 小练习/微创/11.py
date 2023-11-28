import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib import pyplot as plot
from skimage import morphology

# 计算狭窄率，对外膜的mask进行腐蚀操作，去除正常血管壁的厚度
model_name = "label"
result_dir = f"/Volumes/昊大侠/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/短轴动态狭窄率/result/狭窄率图表_0920/results/{model_name}"
file_list = os.listdir(result_dir)
try:
    file_list.remove(".DS_Store")
except ValueError:
    pass
try:
    file_list.remove("._.DS_Store")
except ValueError:
    pass

spacing_pixels = np.array(
    pd.read_excel("/Users/WangHao/Desktop/TODO/Data/动态测试集spacing.xlsx").values
)

pn_all = {}
pne_all = {}
mask_all = {}
for idx, file_name in enumerate(file_list):
    if file_name.endswith(".nii.gz"):
        data = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(result_dir, file_name))
        )
    elif file_name.endswith(".npy"):
        data = np.load(os.path.join(result_dir, file_name), allow_pickle=True)
    else:
        print(f"data format {os.path.splitext(file_name)[-1]} is error")
        sys.exit()

    spacing = int(
        spacing_pixels[np.argwhere(spacing_pixels[:, 0] == file_name[:-7])][0][0][1]
    )

    pn = []
    pne = []
    mask = {}
    mask_out_list = []
    mask_in_list = []
    mask_both_list = []
    mask_both_erode_list = []
    for k, img in enumerate(data):
        # pred 腐蚀
        out_mask = np.zeros_like(img)
        out_mask[img == 1] = 1  # 环形mask
        in_mask = np.zeros_like(img)
        in_mask[img == 2] = 1  # 内膜mask
        both_mask = out_mask + in_mask  # 外膜mask
        both_mask_erode = cv2.erode(
            both_mask, kernel=(3, 3), iterations=spacing // 10
        )  # 外膜腐蚀mask

        mask_out_list.append(out_mask)
        mask_in_list.append(in_mask)
        mask_both_list.append(both_mask)
        mask_both_erode_list.append(both_mask_erode)

        pred_narrow = 1 - (np.sum(in_mask) / np.sum(both_mask))
        pred_erode_narrow = 1 - (np.sum(in_mask) / np.sum(both_mask_erode))

        pn.append(pred_narrow)
        pne.append(pred_erode_narrow)

        if False:
            print(
                f"pred_narrow:{pred_narrow:.2f} | pred_narrow:{pred_erode_narrow:.2f}"
            )

    mask["out"] = mask_out_list
    mask["in"] = mask_in_list
    mask["both"] = mask_both_list
    mask["both_erode"] = mask_both_erode_list

    mask_all[file_name] = mask
    pn_all[file_name] = pd.Series(pn)
    pne_all[file_name] = pd.Series(pne)

if False:
    with pd.ExcelWriter(f"{model_name}.xlsx") as writer:
        df_pn_all = pd.DataFrame(data=pn_all)
        df_pn_all.to_excel(
            writer,
            sheet_name="diameter stenosis",
            index=True,
            header=True,
            startrow=0,
            startcol=0,
        )

    with pd.ExcelWriter(f"{model_name}_erode.xlsx") as writer:
        df_pne_all = pd.DataFrame(data=pne_all)
        df_pne_all.to_excel(
            writer,
            sheet_name="diameter stenosis",
            index=True,
            header=True,
            startrow=0,
            startcol=0,
        )


def remove_isolate(inputs, threshold_area=0.5):
    mask = np.zeros((inputs.shape[0], inputs.shape[1]), dtype=np.uint8)
    mask[np.sum(inputs, axis=-1) > 0] = 1
    mask = morphology.remove_small_objects(
        mask.astype(np.bool8), np.sum(mask) * threshold_area, connectivity=8
    ).astype(np.uint8)
    # outputs = np.expand_dims(mask, -1) * inputs
    outputs = mask * inputs

    return outputs


def minimum_external_circle(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img = np.expand_dims(img, -1).repeat(3, -1)

    cnt = contours[0]
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))  # 最小内接圆圆心
    radius = int(radius)  # 最小内接圆直径
    cv2.circle(img, center, radius, (0, 255, 0), 2)
    cv2.circle(img, center, 1, (0, 255, 0), 2)
    return (x, y), radius, img


# 把笛卡尔坐标转化为极坐标
def to_polar(vector, x, y):
    v_length = np.sqrt((vector[1] - int(x)) ** 2 + (vector[0] - int(y)) ** 2)
    v_angle = np.arctan2(vector[0] - int(y), vector[1] - int(x))
    return (v_length, np.around(v_angle, 2))


def circle_max_distance(polar):
    angle_range = np.linspace(-3.14, 3.14, int((3.14 * 2) / 0.01 + 1))
    polar = np.array(polar)
    polar_angle = polar[:, 1]
    radial = []
    for angle in angle_range:
        idx_list = np.argwhere(polar_angle == np.around(angle, 2))
        if len(idx_list) != 0:
            # if len(idx_list) != 0:
            distance = np.max(polar[idx_list][:, 0, 0]) - np.min(
                polar[idx_list][:, 0, 0]
            )
            radial.append(int(distance))

    return max(radial)


data_distance = []
for i, data_tuple in enumerate(list(mask_all.items())):
    data = data_tuple[1]["out"]
    frame_distance = []
    for j, frame in enumerate(data):
        frame_pad = np.pad(
            frame, ((100, 100), (100, 100)), "constant", constant_values=0
        )
        frame_pad = cv2.morphologyEx(
            frame_pad, cv2.MORPH_CLOSE, kernel=np.ones((15, 15), np.uint8)
        )
        frame = frame_pad[100:-100, 100:-100]
        frame = remove_isolate(frame)
        plot.imshow(frame)
        print(np.max(frame))

        (x, y), radius, visual_frame = minimum_external_circle(frame)
        plot.imshow(visual_frame)

        frame_index = np.argwhere(frame == 1)

        polar = []
        for idx in range(len(frame_index)):
            polar.append(to_polar(frame_index[idx], x, y))
        polar = sorted(polar, key=lambda item: item[1])

        frame_distance.append(circle_max_distance(polar))

    data_distance.append(frame_distance)

print("运行完成")
