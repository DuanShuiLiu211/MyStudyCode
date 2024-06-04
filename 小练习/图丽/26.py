import os
import glob
import argparse
import random
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import shutil


def convert_annotation(img_path, xml_path, out_path):
    class_names = []
    xml_fns = glob.glob(os.path.join(xml_path, "*.xml"))
    print(f"总标注数 [{len(xml_fns)}]。")
    for xml_fn in xml_fns:
        tree = ET.parse(xml_fn)
        root = tree.getroot()
        for obj in root.iter("object"):
            cls = obj.find("name").text  # type: ignore
            class_names.append(cls)
    class_names = sorted(list(set(class_names)))
    print(f"总标注类别 [{class_names}]。")

    images_annotations_dict = {}
    im_fns = glob.glob(os.path.join(img_path, "*.jpg"))
    print(f"总图像数 [{len(im_fns)}]。")
    for im_fn in tqdm(im_fns):
        if os.path.getsize(im_fn) == 0:
            continue
        xml_fn = os.path.join(
            xml_path, os.path.splitext(os.path.basename(im_fn))[0] + ".xml"
        )
        if not os.path.exists(xml_fn):
            continue
        img = Image.open(im_fn)
        height, width = img.height, img.width
        tree = ET.parse(xml_fn)
        root = tree.getroot()
        annotations = []
        xml_height = int(root.find("size").find("height").text)  # type: ignore
        xml_width = int(root.find("size").find("width").text)  # type: ignore
        if height != xml_height or width != xml_width:
            print(
                f"图像 [{im_fn}] 的高宽 [{(height, width)}] 与标注 [{xml_fn}] 的高宽 [{(xml_height, xml_width)}] 不匹配。"
            )
            continue
        for obj in root.iter("object"):
            cls = obj.find("name").text  # type: ignore
            cls_id = class_names.index(cls)
            xmlbox = obj.find("bndbox")
            xmin = int(xmlbox.find("xmin").text)  # type: ignore
            ymin = int(xmlbox.find("ymin").text)  # type: ignore
            xmax = int(xmlbox.find("xmax").text)  # type: ignore
            ymax = int(xmlbox.find("ymax").text)  # type: ignore
            cx = (xmax + xmin) / 2.0 / width
            cy = (ymax + ymin) / 2.0 / height
            bw = (xmax - xmin) * 1.0 / width
            bh = (ymax - ymin) * 1.0 / height
            annotations.append("{} {} {} {} {}".format(cls_id, cx, cy, bw, bh))

        if len(annotations) > 0:
            images_annotations_dict[im_fn] = annotations

    print(f"总图像标注匹配数 [{len(images_annotations_dict)}]。")
    im_fns = list(images_annotations_dict.keys())
    random.seed(51)
    random.shuffle(im_fns)
    train_num = int(len(im_fns) * 0.9)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_path, "labels", split), exist_ok=True)

    for im_fn in tqdm(im_fns[:train_num]):
        shutil.copy(
            im_fn, os.path.join(out_path, "images", "train", os.path.basename(im_fn))
        )
        annotation_txt_filename = os.path.basename(im_fn).replace(".jpg", ".txt")
        annotation_txt_file_path = os.path.join(
            out_path, "labels", "train", annotation_txt_filename
        )
        with open(annotation_txt_file_path, "w") as f:
            f.write("\n".join(images_annotations_dict[im_fn]))

    for im_fn in tqdm(im_fns[train_num:]):
        shutil.copy(
            im_fn, os.path.join(out_path, "images", "val", os.path.basename(im_fn))
        )
        annotation_txt_filename = os.path.basename(im_fn).replace(".jpg", ".txt")
        annotation_txt_file_path = os.path.join(
            out_path, "labels", "val", annotation_txt_filename
        )
        with open(annotation_txt_file_path, "w") as f:
            f.write("\n".join(images_annotations_dict[im_fn]))


def parse_args():
    parser = argparse.ArgumentParser("generate annotation")
    parser.add_argument(
        "--img_path",
        type=str,
        default="/Volumes/tlkj/datas1/import-datas/license-plate-character/images",
        help="input image directory",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default="/Volumes/tlkj/datas1/import-datas/license-plate-character/labels_3cls",
        help="input xml directory",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/Users/WangHao/Sites/学习/LargeData/licence_plate_character/detect",
        help="output directory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    convert_annotation(args.img_path, args.xml_path, args.out_path)
