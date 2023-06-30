import os
import shutil
import argparse


def GetArgs():
    parser = argparse.ArgumentParser(description='将labelme标注后的json文件批量转换为图片')
    parser.add_argument('--input', '-i', help='json文件目录', default='.\\json')
    parser.add_argument('--out-mask',
                        '-m',
                        help='mask图存储目录',
                        default='.\\mask')
    parser.add_argument('--out-img', '-r', help='json文件中提取出的原图存储目录')
    parser.add_argument('--out-viz', '-v', help='mask与原图合并viz图存储目录')
    return parser.parse_args()


if __name__ == '__main__':
    # 调用labelme库中原有的 labelme_json_to_dataset 为核心
    # 批量将文件夹中的 json 文件转换，并抽取对应图片至各自文件夹
    _args = GetArgs()
    _basepath = "/Users/WangHao/Desktop/TODO/Data"
    _args.input = _basepath
    _args.out_mask = f"{_basepath}/mask"
    _args.out_img = f"{_basepath}/img"
    _args.out_viz = f"{_basepath}/viz"

    _jsonFolder = _args.input
    input_files = os.listdir(_jsonFolder)
    for sfn in input_files:  # single file name
        if os.path.splitext(sfn)[1] == ".json":  # 是否为 json 文件

            # 调用 labelme 的 labelme_json_to_dataset 方法执行转换并输出到 temp 文件夹
            os.system(
                f"labelme_json_to_dataset {_jsonFolder}/{sfn} -o {_jsonFolder}/temp"
            )

            # 复制 json 文件中提取出的原图到存储目录
            if _args.out_img:
                if not os.path.exists(_args.out_img):
                    os.makedirs(_args.out_img)

                src_img = f"{_jsonFolder}/temp/img.png"
                dst_img = _args.out_img + '/' + os.path.splitext(
                    sfn)[0] + ".png"
                shutil.copyfile(src_img, dst_img)

            # 复制相应的 mask 图到存储目录
            if _args.out_mask:
                if not os.path.exists(_args.out_mask):
                    os.makedirs(_args.out_mask)

                src_mask = f"{_jsonFolder}/temp/label.png"
                dst_mask = _args.out_mask + '/' + os.path.splitext(
                    sfn)[0] + ".png"
                shutil.copyfile(src_mask, dst_mask)

            # 复制相应的 viz 图到存储目录
            if _args.out_viz:
                if not os.path.exists(_args.out_viz):
                    os.makedirs(_args.out_viz)

                src_viz = f"{_jsonFolder}/temp/label_viz.png"
                dst_viz = _args.out_viz + '/' + os.path.splitext(
                    sfn)[0] + ".png"
                shutil.copyfile(src_viz, dst_viz)
