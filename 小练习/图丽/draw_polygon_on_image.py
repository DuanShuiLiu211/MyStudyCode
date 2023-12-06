import json
import os
import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

image = None
attr_polygons = list()
attr_points = list()
out_polygons = list()
out_points = list()
config_path = None
draw_tag = 0  # 绘制类型 0：识别区域, 1：跟踪区域, 2: lanes

lanes = OrderedDict()
lanes_points = []


def rect_click(event):
    pass


def refresh():
    """
    刷新画布
    """
    w, h = canvas.winfo_width(), canvas.winfo_height()
    canvas.delete(tk.ALL)
    if image is None:
        return
    canvas.create_image(image[0], image[1], anchor=tk.NW, image=image[2])

    for polygon in attr_polygons:
        canvas.create_polygon(polygon, outline="red", fill="")
        canvas.tag_bind("tags", "<Button-1>", rect_click)
        canvas.tag_bind("tags", "<Button-2>", rect_click)
        canvas.tag_bind("tags", "<Button-3>", rect_click)
    for idx in range(len(attr_points) // 2 - 1):
        sx = attr_points[idx * 2]
        sy = attr_points[idx * 2 + 1]
        ex = attr_points[idx * 2 + 2]
        ey = attr_points[idx * 2 + 3]
        canvas.create_line(sx, sy, ex, ey, fill="red")

    for polygon in out_polygons:
        canvas.create_polygon(polygon, outline="blue", fill="")
        canvas.tag_bind("tags", "<Button-1>", rect_click)
        canvas.tag_bind("tags", "<Button-2>", rect_click)
        canvas.tag_bind("tags", "<Button-3>", rect_click)
    for idx in range(len(out_points) // 2 - 1):
        sx = out_points[idx * 2]
        sy = out_points[idx * 2 + 1]
        ex = out_points[idx * 2 + 2]
        ey = out_points[idx * 2 + 3]
        canvas.create_line(sx, sy, ex, ey, fill="blue")

    for polygon in lanes.values():
        canvas.create_polygon(polygon, outline="green", fill="")
        canvas.tag_bind("tags", "<Button-1>", rect_click)
        canvas.tag_bind("tags", "<Button-2>", rect_click)
        canvas.tag_bind("tags", "<Button-3>", rect_click)
    for idx in range(len(lanes_points) // 2 - 1):
        sx = lanes_points[idx * 2]
        sy = lanes_points[idx * 2 + 1]
        ex = lanes_points[idx * 2 + 2]
        ey = lanes_points[idx * 2 + 3]
        canvas.create_line(sx, sy, ex, ey, fill="green")


def refresh_info():
    """
    刷新当前标记框信息
    """
    info_box.delete(0, info_box.size())
    lane_info_box.delete(0, lane_info_box.size())
    canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
    x, y, w, h = image[0], image[1], image[2].width(), image[2].height()
    for rect in attr_polygons + out_polygons:
        info_box.insert(
            tk.END,
            "{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}".format(
                int(rect[0] - x), int(rect[1] - y), int(rect[2] - x), int(rect[3] - y)
            ),
        )
    for _id, _ in lanes.items():
        lane_info_box.insert(tk.END, _id)


def open_image():
    """
    读取图片，同时加载已有检测区域配置
    """
    file_path = filedialog.askopenfile(
        filetypes=[("Image Files", ["*.jpg", "*.jpeg", "*.png"]), ("All Files", "*")]
    )
    if file_path is None:
        return
    file_path = file_path.name

    canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
    frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags=cv2.IMREAD_COLOR)

    x, y = 0, 0
    image_w, image_h = frame.shape[1], frame.shape[0]
    if float(image_w / image_h) > float(canvas_w / canvas_h):
        frame = cv2.resize(frame, (canvas_w, int(canvas_w * image_h / image_w)))
        x = 0
        y = (canvas_h - frame.shape[0]) // 2
    else:
        frame = cv2.resize(frame, (int(canvas_h * image_w / image_h), canvas_h))
        x = (canvas_w - frame.shape[1]) // 2
        y = 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=frame)
    global image
    image = (x, y, img, image_w, image_h)

    attr_polygons.clear()
    out_polygons.clear()
    lanes.clear()
    global config_path
    config_path = filedialog.askopenfile(filetypes=[("Json Files", "*.json")])
    if config_path is None:
        return
    config_path = config_path.name
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

            if (
                "veh_attr_param" in config.keys()
                and "attributing_regions" in config["veh_attr_param"].keys()
            ):
                for region in config["veh_attr_param"]["attributing_regions"]:
                    points = region["ps"]
                    points_ = []
                    for point in points:
                        x_ = x + img.width() * int(point["x"]) / image_w
                        y_ = y + img.height() * int(point["y"]) / image_h
                        points_.extend([x_, y_])

                    attr_polygons.append(points_)

            if (
                "filter_param" in config.keys()
                and "output_regions" in config["filter_param"].keys()
            ):
                for region in config["filter_param"]["output_regions"]:
                    points = region["ps"]
                    points_ = []
                    for point in points:
                        x_ = x + img.width() * int(point["x"]) / image_w
                        y_ = y + img.height() * int(point["y"]) / image_h
                        points_.extend([x_, y_])

                    out_polygons.append(points_)

            if "dt_param" in config.keys() and "lanes" in config["dt_param"].keys():
                for lane in config["dt_param"]["lanes"]:
                    lane_id = lane["id"]
                    points = lane["area"]["ps"]
                    points_ = []
                    for point in points:
                        x_ = x + img.width() * int(point["x"]) / image_w
                        y_ = y + img.height() * int(point["y"]) / image_h
                        points_.extend([x_, y_])
                    lanes[lane_id] = points_
    refresh()
    refresh_info()


def choose_polygon(event):
    # 高亮选择的多边形框
    selection = info_box.curselection()
    if len(selection) == 0:
        return
    polygons = attr_polygons + out_polygons
    polygon = polygons[selection[0]]
    canvas.delete("tmp2")
    if polygon in attr_polygons:
        canvas.create_polygon(polygon, outline="red", fill="red", tags="tmp2")
    else:
        canvas.create_polygon(polygon, outline="blue", fill="blue", tags="tmp2")


def choose_polygon_delete(event):
    """
    选中当前打开的视频的标记框，用于删除
    """

    def delete():
        selection = info_box.curselection()
        if len(selection) == 0:
            return
        positions = info_box.get(selection).split("-")
        positions = [int(pos) for pos in positions]
        canvas.delete("tmp2")
        polygons = attr_polygons + out_polygons
        if polygons[selection[0]] in attr_polygons:
            attr_polygons.remove(polygons[selection[0]])
        else:
            out_polygons.remove(polygons[selection[0]])
        info_box.delete(selection)
        refresh()

    info_box.send
    menu = tk.Menu(main, tearoff=0)
    menu.add_command(label="删除", command=delete)
    menu.post(event.x_root, event.y_root)


def choose_polygon1(event):
    """
    高亮选择的多边形框
    """
    selection = lane_info_box.curselection()
    if len(selection) == 0:
        return
    lane_id = list(lanes)[selection[0]]
    polygon = lanes[lane_id]
    canvas.delete("tmp3")
    canvas.create_polygon(polygon, outline="green", fill="green", tags="tmp3")


def choose_polygon_delete1(event):
    """
    选中当前打开的视频的标记框，用于删除
    """

    def delete():
        selection = lane_info_box.curselection()
        if len(selection) == 0:
            return
        canvas.delete("tmp3")
        lane_id = list(lanes)[selection[0]]
        del lanes[lane_id]
        lane_info_box.delete(selection)
        refresh()

    lane_info_box.send
    menu = tk.Menu(main, tearoff=0)
    menu.add_command(label="删除", command=delete)
    menu.post(event.x_root, event.y_root)


def save():
    """
    保存配置文件
    """
    if image is None:
        return
    x, y, w, h, w_origin, h_origin = (
        image[0],
        image[1],
        image[2].width(),
        image[2].height(),
        image[3],
        image[4],
    )

    result = {}
    with open(config_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    regions = []
    for polygon in attr_polygons:
        points_ = []
        for idx in range(len(polygon) // 2):
            x_ = int((polygon[idx * 2] - x) / w * w_origin)
            y_ = int((polygon[idx * 2 + 1] - y) / h * h_origin)
            points_.append({"x": x_, "y": y_})
        regions.append({"ps": points_})

    result["veh_attr_param"]["attributing_regions"] = regions
    regions = []
    for polygon in out_polygons:
        points_ = []
        for idx in range(len(polygon) // 2):
            x_ = int((polygon[idx * 2] - x) / w * w_origin)
            y_ = int((polygon[idx * 2 + 1] - y) / h * h_origin)
            points_.append({"x": x_, "y": y_})
        regions.append({"ps": points_})

    result["filter_param"]["output_regions"] = regions

    result["dt_param"] = {}
    result["dt_param"]["lanes"] = []
    for id, polygon in lanes.items():
        lane_para = dict()
        lane_para["id"] = id
        points_ = []
        for idx in range(len(polygon) // 2):
            x_ = int((polygon[idx * 2] - x) / w * w_origin)
            y_ = int((polygon[idx * 2 + 1] - y) / h * h_origin)
            points_.append({"x": x_, "y": y_})
        lane_para["area"] = {"ps": points_}

        result["dt_param"]["lanes"].append(lane_para)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def draw_point(event):
    """
    绘制多边形中的单个点
    """
    x, y = event.x, event.y
    if image is None:
        return
    img_x, img_y = (
        image[0],
        image[1],
    )
    img_w, img_h = image[2].width(), image[2].height()
    if x < img_x or x > img_x + img_w or y < img_y or y > img_y + img_h:
        return
    if draw_tag == 0:
        attr_points.extend([x, y])
    elif draw_tag == 1:
        out_points.extend([x, y])
    elif draw_tag == 2:
        lanes_points.extend([x, y])
    else:
        return


def draw_reference_line(event):
    """
    移动鼠标，绘制参考线
    """
    # 清除画布并绘制保存的内容
    refresh()
    canvas.delete("tmp")

    x, y = event.x, event.y
    w, h = canvas.winfo_width(), canvas.winfo_height()
    # 绘制两条直线
    canvas.create_line(0, y, w, y, tags="tmp")
    canvas.create_line(x, 0, x, h, tags="tmp")

    if draw_tag == 0:
        if len(attr_points) == 0:
            return
        x_pre, y_pre = attr_points[-2], attr_points[-1]
        canvas.create_line(x_pre, y_pre, x, y, fill="red", tags="tmp")
    elif draw_tag == 1:
        if len(out_points) == 0:
            return
        x_pre, y_pre = out_points[-2], out_points[-1]
        canvas.create_line(x_pre, y_pre, x, y, fill="blue", tags="tmp")
    elif draw_tag == 2:
        if len(lanes_points) == 0:
            return
        x_pre, y_pre = lanes_points[-2], lanes_points[-1]
        canvas.create_line(x_pre, y_pre, x, y, fill="green", tags="tmp")
    else:
        return


def end_draw(event):
    """
    按 esc 键结束当前绘制
    """
    if draw_tag == 0:
        if len(attr_points) == 0 or len(attr_points) == 1:
            return
        attr_polygons.append(attr_points.copy())
        attr_points.clear()
    elif draw_tag == 1:
        if len(out_points) == 0 or len(out_points) == 1:
            return
        out_polygons.append(out_points.copy())
        out_points.clear()
    elif draw_tag == 2:
        if len(lanes_points) == 0 or len(lanes_points) == 1:
            return
        lane_id = simpledialog.askstring(
            title="获取信息", prompt="输入车道ID", initialvalue="000"
        )
        while lane_id in lanes:
            lane_id += "_"
        lanes[lane_id] = lanes_points.copy()
        lanes_points.clear()
    refresh()
    refresh_info()


def drawpoly1():
    global draw_tag
    draw_tag = 0


def drawpoly2():
    global draw_tag
    draw_tag = 1


def drawpoly3():
    global draw_tag
    draw_tag = 2


# GUI部分
main = tk.Tk()
sw = main.winfo_screenwidth()
sh = main.winfo_screenheight()
ww = 1920
wh = 1080
x = (sw - ww) / 2
y = (sh - wh) / 2
main.title("检测区域配置")
main.geometry("%dx%d+%d+%d" % (ww, wh, x, y))
main.resizable(0, 0)
main.bind_all("<Escape>", end_draw)

# 创建左侧toolbar
toolbar = tk.Frame(main, borderwidth=1, relief=tk.RIDGE)
toolbar.pack(side=tk.LEFT, fill=tk.Y)

btn_openimg = tk.Button(toolbar, text="加载图片与配置", height=3, width=10, command=open_image)
btn_drawpoly1 = tk.Button(toolbar, text="识别区域", height=3, width=10, command=drawpoly1)
btn_drawpoly2 = tk.Button(toolbar, text="跟踪区域", height=3, width=10, command=drawpoly2)
btn_drawpoly3 = tk.Button(toolbar, text="车道区域", height=3, width=10, command=drawpoly3)
btn_save = tk.Button(toolbar, text="保存参数", height=3, width=10, command=save)

btn_openimg.pack(side=tk.TOP)
btn_drawpoly1.pack(side=tk.TOP)
btn_drawpoly2.pack(side=tk.TOP)
btn_drawpoly3.pack(side=tk.TOP)
btn_save.pack(side=tk.TOP)

# 右侧的canvas
canvas = tk.Canvas(main, borderwidth=1, relief=tk.RIDGE)
canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas.bind("<Button-1>", draw_point)
canvas.bind("<Motion>", draw_reference_line)

# 底部的infobox
info_box = tk.Listbox(main, height=10)
info_box.pack(side=tk.BOTTOM, fill=tk.X)
info_box.bind("<Button-1>", choose_polygon)
info_box.bind("<Button-3>", choose_polygon_delete)

lane_info_box = tk.Listbox(info_box, width=50)
lane_info_box.pack(side=tk.RIGHT, fill=tk.Y)
lane_info_box.bind("<Button-1>", choose_polygon1)
lane_info_box.bind("<Button-3>", choose_polygon_delete1)

main.mainloop()
