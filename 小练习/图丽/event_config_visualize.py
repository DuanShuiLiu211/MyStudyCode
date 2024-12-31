import argparse
import json

import matplotlib.patches as patches
import matplotlib.pyplot as plot
from matplotlib.font_manager import FontProperties
from PIL import Image


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="可视化显示ROI和车道线数据")
    parser.add_argument(
        "--base_path", type=str, required=True, help="JSON和图片文件的基础路径"
    )
    parser.add_argument(
        "--name_mark", type=str, required=True, help="JSON和图片文件的标识名"
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="./assets/font/simhei.ttf",
        help="中文字体文件路径",
    )
    return parser.parse_args()


def load_data(file_path):
    """从文件加载JSON数据"""
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def load_image(image_path):
    """加载图片文件"""
    image = Image.open(image_path)
    return image


def plot_roi(ax, rois, font_prop):
    """绘制感兴趣区域(ROIs)

    参数:
        ax: matplotlib轴对象
        rois: ROI数据列表
        font_prop: 字体属性
    """
    for roi in rois:
        area_points = roi["cover_area"]["ps"]
        polygon = patches.Polygon(
            [(p["x"], p["y"]) for p in area_points],
            closed=True,
            fill=False,
            edgecolor="r",
            linewidth=2,
        )
        ax.add_patch(polygon)
        plot.text(
            area_points[0]["x"],
            area_points[0]["y"],
            f"{roi['name']}_{roi['description']}",
            color="r",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
            fontproperties=font_prop,
        )


def plot_roi_lines(roi_lines, font_prop):
    """绘制感兴趣线条(ROI lines)

    参数:
        roi_lines: ROI线条数据列表
        font_prop: 字体属性
    """
    for line in roi_lines:
        p1 = line["line"]["p1"]
        p2 = line["line"]["p2"]
        plot.plot([p1["x"], p2["x"]], [p1["y"], p2["y"]], "g-", linewidth=2)
        mid_x = (p1["x"] + p2["x"]) / 2
        mid_y = (p1["y"] + p2["y"]) / 2
        plot.text(
            mid_x,
            mid_y,
            f"{line['name']}_{line['description']}",
            color="g",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
            fontproperties=font_prop,
        )


def plot_lane(ax, lanes, font_prop):
    """绘制车道线

    参数:
        ax: matplotlib轴对象
        lanes: 车道线数据列表
        font_prop: 字体属性
    """
    for lane in lanes:
        area_points = lane["cover_area"]["ps"]
        polygon = patches.Polygon(
            [(p["x"], p["y"]) for p in area_points],
            closed=True,
            fill=False,
            edgecolor="r",
            linewidth=2,
        )
        ax.add_patch(polygon)
        plot.text(
            area_points[0]["x"],
            area_points[0]["y"],
            f"{lane['lane_id']}_{lane['max_speed']}",
            color="r",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
            fontproperties=font_prop,
        )


def visualize_data(image_path, data, font_path):
    """在背景图像上可视化显示ROIs和ROI线条

    参数:
        image_path: 背景图片路径
        data: JSON数据
        font_path: 字体文件路径
    """
    # 加载背景图片
    image = load_image(image_path)
    _, ax = plot.subplots()
    ax.imshow(image)

    # 设置中文字体属性
    font_prop = FontProperties(fname=font_path)

    # 绘制ROIs、ROI线条和车道线
    if data.get("rois"):
        plot_roi(ax, data["rois"], font_prop)
    if data.get("roi_lines"):
        plot_roi_lines(data["roi_lines"], font_prop)
    if data.get("lanes"):
        plot_lane(ax, data["lanes"], font_prop)

    # 设置图表参数
    plot.title("目标区域可视化", fontproperties=font_prop)
    plot.xlabel("X坐标", fontproperties=font_prop)
    plot.ylabel("Y坐标", fontproperties=font_prop)
    plot.grid(True)

    # 显示图表
    plot.show()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()

    # 构建文件路径
    json_file_path = f"{args.base_path}/{args.name_mark}.json"
    image_file_path = f"{args.base_path}/{args.name_mark}.jpg"

    # 加载数据并可视化
    data = load_data(json_file_path)
    visualize_data(image_file_path, data, args.font_path)
