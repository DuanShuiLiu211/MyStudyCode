import json
import os
import sys

import cv2
import numpy as np


def adjust_pts_order(pts_2ds):
    """sort rectangle points by counterclockwise"""

    cen_x, cen_y = np.mean(pts_2ds, axis=0)
    # refer_line = np.array([10,0])

    d2s = []
    for i in range(len(pts_2ds)):
        o_x = pts_2ds[i][0] - cen_x
        o_y = pts_2ds[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([pts_2ds[i], atan2])

    d2s = sorted(d2s, key=lambda x: x[1])
    order_2ds = np.array([x[0] for x in d2s])

    return order_2ds


if __name__ == "__main__":
    img_folder = r"/Users/WangHao/Desktop/UpData/迪-静脉/image"
    label_folder = r"/Users/WangHao/Desktop/UpData/迪-静脉/json"
    save_folder = r"/Users/WangHao/Desktop/UpData/迪-静脉/dataset"

    img_list = os.listdir(img_folder)
    label_list = os.listdir(label_folder)

    for i in range(len(label_list)):
        img = img_list[i]
        basename_img = str.split(img, ".")[0]

        label = f"{basename_img}.json"

        img_dir = os.path.join(img_folder, img)
        label_dir = os.path.join(label_folder, label)

        if label not in label_list:
            raise ValueError(f"图片{img} and 标注{label} not match")

        cv_img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)
        if cv_img.shape[-1] == 4:
            cv_img = cv_img[..., 0:-1]

        with open(label_dir, "r") as f:
            dict = json.load(f)

        shapes = dict["shapes"]

        outpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)
        inpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)
        vein_points = np.ndarray(shape=(0, 2), dtype=np.uint8)

        for j in range(len(shapes)):
            kind = shapes[j]["label"]
            x = int(shapes[j]["points"][0][0])
            y = int(shapes[j]["points"][0][1])
            point = np.asarray([x, y])

            if kind == "1":
                outpoints = np.vstack((outpoints, point))
            elif kind == "2":
                inpoints = np.vstack((inpoints, point))
            elif kind == "3":
                vein_points = np.vstack((vein_points, point))
            else:
                raise ValueError(f"point label error: {kind}")

        if len(cv_img.shape) == 3:
            if cv_img.shape[-1] == 3:
                label_img = np.zeros_like(cv_img)
            elif cv_img.shape[-1] == 4:
                label_img = np.zeros_like(cv_img)[..., 0:-1]
            else:
                print("输入图片格式不是RGB也不是RGBA")
                sys.exit()
        else:
            cv_img = np.expand_dims(cv_img, axis=-1).repeat(axis=-1, repeats=3)
            label_img = np.zeros_like(cv_img)

        # 填充

        cv2.fillPoly(label_img, pts=[adjust_pts_order(outpoints)], color=(0, 0, 255))
        cv2.fillPoly(label_img, pts=[adjust_pts_order(inpoints)], color=(0, 255, 0))
        if np.sum(label_img[..., 0] + label_img[..., 2]) == 0:
            cv2.fillPoly(label_img, pts=[adjust_pts_order(inpoints)], color=(0, 0, 255))
            cv2.fillPoly(
                label_img, pts=[adjust_pts_order(outpoints)], color=(0, 255, 0)
            )
        if vein_points.shape[0] != 0:
            cv2.fillPoly(
                label_img, pts=[adjust_pts_order(vein_points)], color=(255, 0, 0)
            )

        save_dir = os.path.join(save_folder, f"迪静脉_{i}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imencode(".png", cv_img)[1].tofile(os.path.join(save_dir, "image.png"))
        # from PIL import Image
        # im = Image.fromarray(label_img[..., 0])
        # im.save(os.path.join(save_dir, 'label0.png'))
        # im = Image.fromarray(label_img[..., 1])
        # im.save(os.path.join(save_dir, 'label1.png'))
        # im = Image.fromarray(label_img[..., 2])
        # im.save(os.path.join(save_dir, 'label2.png'))
        cv2.imencode(".png", label_img)[1].tofile(os.path.join(save_dir, "label.png"))
