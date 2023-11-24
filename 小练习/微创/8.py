import json
import os

import cv2
import numpy as np

if __name__ == "__main__":
    # 纵切
    img_folder = (
        r"D:\lessons\实习相关\微创实习\dataCvtCode\dataFolder\data-0718-长轴\长轴静态斑块标注--迪-image"
    )
    label_folder = (
        r"D:\lessons\实习相关\微创实习\dataCvtCode\dataFolder\data-0718-长轴\长轴静态斑块标注--迪-json"
    )
    visual_folder = r"D:\lessons\实习相关\微创实习\0704数据\visual"
    save_folder = (
        r"D:\lessons\实习相关\微创实习\dataCvtCode\dataFolder\data-0718-长轴\长轴静态斑块标注--迪-dataset"
    )

    img_list = os.listdir(img_folder)
    label_list = os.listdir(label_folder)

    for i in range(len(label_list)):
        img = label_list[i].replace(".json", ".jpg")

        label = label_list[i]

        img_dir = os.path.join(img_folder, img)
        label_dir = os.path.join(label_folder, label)

        basename = str.split(img, ".")[0]
        basename2 = str.split(label, ".")[0]

        if basename != basename2:
            raise ValueError(f"{basename}.jpg and {basename2}.json not match")

        cv_img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)

        with open(label_dir, "r") as f:
            dict = json.load(f)

        shapes = dict["shapes"]

        MAB_points_1 = np.ndarray(shape=(0, 2), dtype=np.uint8)
        lumen_points_1 = np.ndarray(shape=(0, 2), dtype=np.uint8)
        MAB_points_2 = np.ndarray(shape=(0, 2), dtype=np.uint8)
        lumen_points_2 = np.ndarray(shape=(0, 2), dtype=np.uint8)

        for i in range(len(shapes)):
            kind = shapes[i]["label"]
            x = int(shapes[i]["points"][0][0])
            y = int(shapes[i]["points"][0][1])
            point = np.asarray([x, y])

            if kind == "1":
                MAB_points_1 = np.vstack((MAB_points_1, point))
            elif kind == "2":
                lumen_points_1 = np.vstack((lumen_points_1, point))
            elif kind == "3":
                lumen_points_2 = np.vstack((lumen_points_2, point))
            elif kind == "4":
                MAB_points_2 = np.vstack((MAB_points_2, point))
            else:
                raise ValueError(f"point label error: {kind}")

        MAB_arroutpoints1 = np.lexsort((MAB_points_1[:, 1], MAB_points_1[:, 0]))
        MAB_points_1 = MAB_points_1[MAB_arroutpoints1, :]
        MAB_arroutpoints2 = np.lexsort((MAB_points_2[:, 1], MAB_points_2[:, 0]))
        MAB_points_2 = MAB_points_2[MAB_arroutpoints2, :]
        lumen_arroutpoints1 = np.lexsort((lumen_points_1[:, 1], lumen_points_1[:, 0]))
        lumen_points_1 = lumen_points_1[lumen_arroutpoints1, :]
        lumen_arroutpoints2 = np.lexsort((lumen_points_2[:, 1], lumen_points_2[:, 0]))
        lumen_points_2 = lumen_points_2[lumen_arroutpoints2, :]

        if len(cv_img.shape) == 3:
            label_img = np.zeros_like(cv_img)
        else:
            cv_img = np.expand_dims(cv_img, axis=-1).repeat(axis=-1, repeats=3)
            label_img = np.zeros_like(cv_img)

        # 填充
        outpoints2 = np.flipud(MAB_points_2)
        inpoints2 = np.flipud(lumen_points_2)

        inpoints = np.vstack((MAB_points_1, outpoints2))
        outpoints = np.vstack((lumen_points_1, inpoints2))
        cv2.fillPoly(label_img, pts=[(inpoints)], color=(0, 0, 255))
        cv2.fillPoly(label_img, pts=[(outpoints)], color=(0, 255, 0))

        if np.sum(label_img[..., 0] + label_img[..., 2]) == 0:
            cv2.fillPoly(label_img, pts=[(inpoints)], color=(0, 0, 255))
            cv2.fillPoly(label_img, pts=[(outpoints)], color=(0, 255, 0))

        save_dir = os.path.join(save_folder, basename)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imencode(".png", cv_img)[1].tofile(os.path.join(save_dir, "image.png"))
        cv2.imencode(".png", label_img)[1].tofile(os.path.join(save_dir, "label.png"))
