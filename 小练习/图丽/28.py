import numpy as np
import cv2
import os


cache_file_path = "train.cache"
save_dir = "train"
os.makedirs(save_dir, exist_ok=True)

try:
    cache_data = np.load(cache_file_path, allow_pickle=True).item()
    for i, label_info in enumerate(cache_data["labels"]):
        img_path = label_info["im_file"]
        img_path = img_path.replace(
            "/home/tlkj/datas/wangh/license-plate-character/images",
            "/Users/WangHao/Sites/学习/LargeData/licence_plate_character/detect/images",
        )
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image {img_path}")
            continue

        img_height, img_width = img.shape[:2]

        for j, bbox in enumerate(label_info["bboxes"]):
            x_center, y_center, width, height = bbox
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            if label_info["cls"][j] == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
            elif label_info["cls"][j] == 1:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
            elif label_info["cls"][j] == 2:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
            else:
                print(f"Failed to draw rectangle {img_path}")
                continue

        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
except Exception as e:
    print(f"An error occurred: {e}")
