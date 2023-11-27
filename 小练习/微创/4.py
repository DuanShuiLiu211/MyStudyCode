import json
import os

import cv2
import numpy as np

if __name__ == "__main__":
    folder_name = r"2--长轴静态-HSF"
    data_folder = f"/Users/WangHao/Desktop/TODO/Data/{folder_name}"
    save_folder = f"/Users/WangHao/Desktop/TODO/Data/{folder_name}/dataset"

    data_list = os.listdir(data_folder)
    try:
        data_list.remove(".DS_Store")
    except:
        pass
    try:
        data_list.remove("._.DS_Store")
    except:
        pass
    try:
        data_list.remove("dataset")
    except:
        pass

    for i in range(len(data_list)):
        data_img_label = os.listdir(os.path.join(data_folder, data_list[i]))
        try:
            data_img_label.remove(".DS_Store")
        except:
            pass
        try:
            data_img_label.remove("._.DS_Store")
        except:
            pass
        if len(data_img_label) <= 4:
            flag = 0
            for filename in data_img_label:
                if ".json" in filename:
                    flag = 1
                    break
            if flag == 1:
                for filename in data_img_label:
                    if ".json" in filename:
                        label = filename
                    if "image" in filename and ".json" not in filename:
                        image = filename
            else:
                for filename in data_img_label:
                    if "label" in filename:
                        label = filename
                    if "image" in filename:
                        image = filename

            img_dir = os.path.join(data_folder, data_list[i], image)
            cv_img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)
            label_dir = os.path.join(data_folder, data_list[i], label)

            if flag == 1:
                with open(label_dir, "r") as f:
                    dict = json.load(f)

                shapes = dict["shapes"]

                front_outpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)
                front_inpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)
                back_outpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)
                back_inpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)

                for j in range(len(shapes)):
                    kind = shapes[j]["label"]
                    x = int(shapes[j]["points"][0][0])
                    y = int(shapes[j]["points"][0][1])
                    point = np.asarray([x, y])

                    if kind == "1":
                        front_outpoints = np.vstack((front_outpoints, point))
                    elif kind == "2":
                        front_inpoints = np.vstack((front_inpoints, point))
                    elif kind == "3":
                        back_inpoints = np.vstack((back_inpoints, point))
                    elif kind == "4":
                        back_outpoints = np.vstack((back_outpoints, point))
                    else:
                        raise ValueError(f"point label error: {kind}")

                front_outpoints = np.sort(front_outpoints, 0)
                front_inpoints = np.sort(front_inpoints, 0)
                back_outpoints = np.sort(back_outpoints, 0)
                back_inpoints = np.sort(back_inpoints, 0)

                point_list_r1 = np.vstack((front_inpoints[::-1], front_outpoints))
                point_list_g1 = np.vstack((back_inpoints[::-1], front_inpoints))
                point_list_r2 = np.vstack((back_inpoints[::-1], back_outpoints))

                if len(cv_img.shape) == 3:
                    label_img = np.zeros_like(cv_img)
                else:
                    cv_img = np.expand_dims(cv_img, axis=-1).repeat(axis=-1, repeats=3)
                    label_img = np.zeros_like(cv_img)

                # 填充
                cv2.fillPoly(label_img, pts=[point_list_r1], color=(0, 0, 255))  # R
                cv2.fillPoly(label_img, pts=[point_list_g1], color=(0, 255, 0))  # G
                cv2.fillPoly(label_img, pts=[point_list_r2], color=(0, 0, 255))  # R
                # import matplotlib.pyplot as plot
                # plot.figure(1)
                # plot.imshow(cv_img)
                # plot.imshow(label_img)
                # plot.show()

                label_img = cv2.flip(label_img, 1)

                img_add = cv2.addWeighted(cv_img, 0.7, label_img, 0.3, 0)

                save_dir = os.path.join(save_folder, f"{folder_name}_{data_list[i]}")

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                cv2.imencode(".png", img_add)[1].tofile(
                    os.path.join(save_dir, "add.png")
                )

            else:
                label_img = cv2.imdecode(np.fromfile(label_dir, dtype=np.uint8), -1)

            save_dir = os.path.join(save_folder, f"{folder_name}_{data_list[i]}")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            cv2.imencode(".png", cv_img)[1].tofile(os.path.join(save_dir, "image.png"))
            cv2.imencode(".png", label_img)[1].tofile(
                os.path.join(save_dir, "label.png")
            )

        else:
            labels = []
            labels_1 = []
            labels_2 = []
            images = []

            for filename in data_img_label:
                if ".json" in filename:
                    labels_1.append(filename)
                if "label" in filename:
                    labels_2.append(filename)
                if "image" in filename and ".json" not in filename:
                    images.append(filename)

            if len(labels_1) == 2:
                labels = labels_1
            elif len(labels_1) == 1:
                numflag = labels_1[0].split(".")[0][-1:]
                for name in labels_2:
                    if numflag in name:
                        labels.append(labels_1[0])
                    else:
                        labels.append(name)
            else:
                labels = labels_2

            assert len(labels) == len(images)

            for k in range(len(labels)):
                label = labels[k]
                flag = 0
                if ".json" in label:
                    flag = 1
                for name in images:
                    if label.split(".")[0][5:] == name.split(".")[0][5:]:
                        image = name

                img_dir = os.path.join(data_folder, data_list[i], image)
                cv_img = cv2.imdecode(np.fromfile(img_dir, dtype=np.uint8), -1)
                label_dir = os.path.join(data_folder, data_list[i], label)

                if flag == 1:
                    with open(label_dir, "r") as f:
                        dict = json.load(f)

                    shapes = dict["shapes"]

                    front_outpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)
                    front_inpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)
                    back_outpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)
                    back_inpoints = np.ndarray(shape=(0, 2), dtype=np.uint8)

                    for j in range(len(shapes)):
                        kind = shapes[j]["label"]
                        x = int(shapes[j]["points"][0][0])
                        y = int(shapes[j]["points"][0][1])
                        point = np.asarray([x, y])

                        if kind == "1":
                            front_outpoints = np.vstack((front_outpoints, point))
                        elif kind == "2":
                            front_inpoints = np.vstack((front_inpoints, point))
                        elif kind == "3":
                            back_inpoints = np.vstack((back_inpoints, point))
                        elif kind == "4":
                            back_outpoints = np.vstack((back_outpoints, point))
                        else:
                            raise ValueError(f"point label error: {kind}")

                    front_outpoints = np.sort(front_outpoints, 0)
                    front_inpoints = np.sort(front_inpoints, 0)
                    back_outpoints = np.sort(back_outpoints, 0)
                    back_inpoints = np.sort(back_inpoints, 0)

                    point_list_r1 = np.vstack((front_inpoints[::-1], front_outpoints))
                    point_list_g1 = np.vstack((back_inpoints[::-1], front_inpoints))
                    point_list_r2 = np.vstack((back_inpoints[::-1], back_outpoints))

                    if len(cv_img.shape) == 3:
                        label_img = np.zeros_like(cv_img)
                    else:
                        cv_img = np.expand_dims(cv_img, axis=-1).repeat(
                            axis=-1, repeats=3
                        )
                        label_img = np.zeros_like(cv_img)

                    # 填充
                    cv2.fillPoly(label_img, pts=[point_list_r1], color=(0, 0, 255))  # R
                    cv2.fillPoly(label_img, pts=[point_list_g1], color=(0, 255, 0))  # G
                    cv2.fillPoly(label_img, pts=[point_list_r2], color=(0, 0, 255))  # R
                    # import matplotlib.pyplot as plot
                    # plot.imshow(cv_img)
                    # plot.imshow(label_img)
                    # plot.show()

                    label_img = cv2.flip(label_img, 1)

                    img_add = cv2.addWeighted(cv_img, 0.7, label_img, 0.3, 0)

                    save_dir = os.path.join(
                        save_folder, f"{folder_name}_{data_list[i]}"
                    )

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    cv2.imencode(".png", img_add)[1].tofile(
                        os.path.join(save_dir, "add.png")
                    )

                else:
                    label_img = cv2.imdecode(np.fromfile(label_dir, dtype=np.uint8), -1)

                save_dir = os.path.join(
                    save_folder, f"{folder_name}_{data_list[i]}_{k}"
                )

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                cv2.imencode(".png", cv_img)[1].tofile(
                    os.path.join(save_dir, "image.png")
                )
                cv2.imencode(".png", label_img)[1].tofile(
                    os.path.join(save_dir, "label.png")
                )
