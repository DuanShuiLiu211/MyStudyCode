import json
import os

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
    # 横切
    img_folder = r"D:\lessons\实习相关\微创实习\dataCvtCode\dataFolder\data-0718\image"
    label_folder = r"D:\lessons\实习相关\微创实习\dataCvtCode\dataFolder\data-0718\json"
    save_folder = (
        r"D:\lessons\实习相关\微创实习\dataCvtCode\dataFolder\data-0718\dataset"
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

        CCA_lumen_points = np.ndarray(shape=(0, 2), dtype=np.uint8)
        CCA_MAB_points = np.ndarray(shape=(0, 2), dtype=np.uint8)

        IJV_points = np.ndarray(shape=(0, 2), dtype=np.uint8)

        BIF_lumen_points = np.ndarray(shape=(0, 2), dtype=np.uint8)
        BIF_MAB_points = np.ndarray(shape=(0, 2), dtype=np.uint8)

        ECA_lumen_points = np.ndarray(shape=(0, 2), dtype=np.uint8)
        ECA_MAB_points = np.ndarray(shape=(0, 2), dtype=np.uint8)

        ICA_lumen_points = np.ndarray(shape=(0, 2), dtype=np.uint8)
        ICA_MAB_points = np.ndarray(shape=(0, 2), dtype=np.uint8)

        for i in range(len(shapes)):
            kind = shapes[i]["label"]
            x = int(shapes[i]["points"][0][0])
            y = int(shapes[i]["points"][0][1])
            point = np.asarray([x, y])

            if kind == "1":
                CCA_lumen_points = np.vstack((CCA_lumen_points, point))
            elif kind == "2":
                CCA_MAB_points = np.vstack((CCA_MAB_points, point))
            elif kind == "3":
                IJV_points = np.vstack((IJV_points, point))
            elif kind == "4":
                BIF_lumen_points = np.vstack((BIF_lumen_points, point))
            elif kind == "5":
                BIF_MAB_points = np.vstack((BIF_MAB_points, point))
            elif kind == "6":
                ICA_lumen_points = np.vstack((ICA_lumen_points, point))
            elif kind == "7":
                ICA_MAB_points = np.vstack((ICA_MAB_points, point))
            elif kind == "8":
                ECA_lumen_points = np.vstack((ECA_lumen_points, point))
            elif kind == "9":
                ECA_MAB_points = np.vstack((ECA_MAB_points, point))
            else:
                raise ValueError(f"point label error: {kind}")

        if len(cv_img.shape) == 3:
            label_img = np.zeros_like(cv_img)
        else:
            cv_img = np.expand_dims(cv_img, axis=-1).repeat(axis=-1, repeats=3)
            label_img = np.zeros_like(cv_img)

        if cv_img.shape[-1] == 4:
            cv_img = cv_img[..., 0:3]
            label_img = np.zeros_like(cv_img)

        if (len(CCA_MAB_points) != 0) & (len(CCA_lumen_points) != 0):
            cv2.fillPoly(
                label_img, pts=[adjust_pts_order(CCA_MAB_points)], color=(0, 0, 255)
            )
            cv2.fillPoly(
                label_img, pts=[adjust_pts_order(CCA_lumen_points)], color=(0, 255, 0)
            )

        if len(IJV_points) != 0:
            cv2.fillPoly(
                label_img, pts=[adjust_pts_order(IJV_points)], color=(255, 0, 0)
            )

        if (len(BIF_lumen_points) != 0) & (len(BIF_MAB_points) != 0):
            cv2.fillPoly(
                label_img, pts=[adjust_pts_order(BIF_MAB_points)], color=(16, 16, 16)
            )
            cv2.fillPoly(
                label_img,
                pts=[adjust_pts_order(BIF_lumen_points)],
                color=(16 * 2, 16 * 2, 16 * 2),
            )

        if (len(ICA_lumen_points) != 0) & (len(ICA_MAB_points) != 0):
            cv2.fillPoly(
                label_img,
                pts=[adjust_pts_order(ICA_MAB_points)],
                color=(16 * 3, 16 * 3, 16 * 3),
            )
            cv2.fillPoly(
                label_img,
                pts=[adjust_pts_order(ICA_lumen_points)],
                color=(16 * 4, 16 * 4, 16 * 4),
            )

        if (len(ECA_lumen_points) != 0) & (len(ECA_MAB_points) != 0):
            cv2.fillPoly(
                label_img,
                pts=[adjust_pts_order(ECA_MAB_points)],
                color=(16 * 5, 16 * 5, 16 * 5),
            )
            cv2.fillPoly(
                label_img,
                pts=[adjust_pts_order(ECA_lumen_points)],
                color=(16 * 6, 16 * 6, 16 * 6),
            )

        save_dir = os.path.join(save_folder, basename)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imencode(".png", cv_img)[1].tofile(os.path.join(save_dir, "image.png"))
        cv2.imencode(".png", label_img)[1].tofile(os.path.join(save_dir, "label.png"))
