import cv2
import os
import json
import tkinter as tk
import numpy as np
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
from collections import OrderedDict


image = None
config_path = None
draw_tag = 0  # 绘制类型 0：lanes 1：rois 2：widths
lanes = OrderedDict()
lanes_points = []
rois = OrderedDict()
rois_points = []
widths = OrderedDict()
width_points = []


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

    for polygon in lanes.values():
        canvas.create_polygon(polygon, outline="red", fill="")
        canvas.tag_bind("tags", "<Button-1>", rect_click)
        canvas.tag_bind("tags", "<Button-2>", rect_click)
        canvas.tag_bind("tags", "<Button-3>", rect_click)
    for idx in range(len(lanes_points) // 2 - 1):
        sx = lanes_points[idx * 2]
        sy = lanes_points[idx * 2 + 1]
        ex = lanes_points[idx * 2 + 2]
        ey = lanes_points[idx * 2 + 3]
        canvas.create_line(sx, sy, ex, ey, fill="red")

    for polygon in rois.values():
        canvas.create_polygon(polygon, outline="blue", fill="")
        canvas.tag_bind("tags", "<Button-1>", rect_click)
        canvas.tag_bind("tags", "<Button-2>", rect_click)
        canvas.tag_bind("tags", "<Button-3>", rect_click)
    for idx in range(len(rois_points) // 2 - 1):
        sx = rois_points[idx * 2]
        sy = rois_points[idx * 2 + 1]
        ex = rois_points[idx * 2 + 2]
        ey = rois_points[idx * 2 + 3]
        canvas.create_line(sx, sy, ex, ey, fill="blue")

    for line in widths.values():
        sx, sy, ex, ey = line
        canvas.create_line(sx, sy, ex, ey, fill="green")


def refresh_info():
    """
    刷新当前标记框信息
    """
    info_box.delete(0, info_box.size())
    roi_info_box.delete(0, roi_info_box.size())
    for id, _ in lanes.items():
        info_box.insert(tk.END, id)
    for id, _ in rois.items():
        roi_info_box.insert(tk.END, id)


def open_image():
    """
    读取图片，同时加载已有检测区域配置
    """
    file_path = filedialog.askopenfile(
        filetypes=[
            ("JPG Files", "*.jpg"),
            ("JPEG Files", "*.jpeg"),
            ("PNG Files", "*.png"),
        ]
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

    lanes.clear()
    rois.clear()
    widths.clear()
    global config_path
    config_path = filedialog.askopenfile(filetypes=[("Json Files", "*.json")])
    if config_path is None:
        return
    config_path = config_path.name
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

            for lane in config["lanes"]:
                lane_id = lane["lane_id"]
                points = lane["cover_area"]["ps"]
                points_ = []
                for point in points:
                    x_ = x + img.width() * int(point["x"]) / image_w
                    y_ = y + img.height() * int(point["y"]) / image_h
                    points_.extend([x_, y_])
                lanes[lane_id] = points_

            for region in config["rois"]:
                name = region["name"]
                points = region["cover_area"]["ps"]
                points_ = []
                for point in points:
                    x_ = x + img.width() * int(point["x"]) / image_w
                    y_ = y + img.height() * int(point["y"]) / image_h
                    points_.extend([x_, y_])
                rois[name] = points_

            for width in config.get("widths", []):
                lane_id = width["lane_id"]
                points = width["points"]
                points_ = []
                for point in points:
                    x_ = x + img.width() * int(point["x"]) / image_w
                    y_ = y + img.height() * int(point["y"]) / image_h
                    points_.extend([x_, y_])
                widths[lane_id] = points_
    refresh()
    refresh_info()


def choose_polygon(event):
    # 高亮选择的多边形框
    selection = info_box.curselection()
    if len(selection) == 0:
        return
    lane_id = list(lanes)[selection[0]]
    polygon = lanes[lane_id]
    canvas.delete("tmp2")
    canvas.create_polygon(polygon, outline="red", fill="red", tags="tmp2")


def choose_polygon_delete(event):
    """
    选中当前打开的视频的标记框，用于删除
    """

    def delete():
        selection = info_box.curselection()
        if len(selection) == 0:
            return
        canvas.delete("tmp2")
        lane_id = list(lanes)[selection[0]]
        del lanes[lane_id]
        info_box.delete(selection)
        refresh()

    info_box.send
    menu = tk.Menu(main, tearoff=0)
    menu.add_command(label="删除", command=delete)
    menu.post(event.x_root, event.y_root)


def choose_polygon1(event):
    # 高亮选择的多边形框
    selection = roi_info_box.curselection()
    if len(selection) == 0:
        return
    lane_id = list(rois)[selection[0]]
    polygon = rois[lane_id]
    canvas.delete("tmp2")
    canvas.create_polygon(polygon, outline="blue", fill="blue", tags="tmp2")


def choose_polygon_delete1(event):
    """
    选中当前打开的视频的标记框，用于删除
    """

    def delete():
        selection = roi_info_box.curselection()
        if len(selection) == 0:
            return
        canvas.delete("tmp2")
        lane_id = list(rois)[selection[0]]
        del rois[lane_id]
        roi_info_box.delete(selection)
        refresh()

    roi_info_box.send
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
    with open(file=str(config_path), mode="r", encoding="utf-8") as f:
        result = json.load(f)
    result["lanes"] = []
    for id, polygon in lanes.items():
        lane_para = dict()
        lane_para["lane_id"] = id
        lane_para["max_speed"] = 100
        lane_para["should_estimate_width"] = True
        lane_para["lane_width"] = 2.8  # 假设固定宽度，也可以改为动态输入
        if id in widths:
            line = widths[id]
            p1x = int((line[0] - x) / w * w_origin)
            p1y = int((line[1] - y) / h * h_origin)
            p2x = int((line[2] - x) / w * w_origin)
            p2y = int((line[3] - y) / h * h_origin)
            lane_para["detect_line"] = {
                "p1": {"x": p1x, "y": p1y},
                "p2": {"x": p2x, "y": p2y},
            }

        points_ = []
        for idx in range(len(polygon) // 2):
            x_ = int((polygon[idx * 2] - x) / w * w_origin)
            y_ = int((polygon[idx * 2 + 1] - y) / h * h_origin)
            points_.append({"x": x_, "y": y_})
        lane_para["cover_area"] = {"ps": points_}

        result["lanes"].append(lane_para)

    result["rois"] = []
    for id, polygon in rois.items():
        roi_para = dict()
        roi_para["name"] = id
        roi_para["description"] = ""
        points_ = []
        for idx in range(len(polygon) // 2):
            x_ = int((polygon[idx * 2] - x) / w * w_origin)
            y_ = int((polygon[idx * 2 + 1] - y) / h * h_origin)
            points_.append({"x": x_, "y": y_})
        roi_para["cover_area"] = {"ps": points_}

        result["rois"].append(roi_para)

    with open(file=str(config_path), mode="w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


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
        lanes_points.extend([x, y])
    elif draw_tag == 1:
        rois_points.extend([x, y])
    elif draw_tag == 2:
        width_points.extend([x, y])
        if len(width_points) == 4:
            lane_id = simpledialog.askstring(
                title="获取信息", prompt="输入车道ID", initialvalue="000"
            )
            while lane_id in widths:
                lane_id += "_"
            widths[lane_id] = width_points.copy()
            width_points.clear()
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
        if len(lanes_points) == 0:
            return
        x_pre, y_pre = lanes_points[-2], lanes_points[-1]
        canvas.create_line(x_pre, y_pre, x, y, fill="red", tags="tmp")
    elif draw_tag == 1:
        if len(rois_points) == 0:
            return
        x_pre, y_pre = rois_points[-2], rois_points[-1]
        canvas.create_line(x_pre, y_pre, x, y, fill="blue", tags="tmp")
    elif draw_tag == 2:
        if len(width_points) == 2:
            x_pre, y_pre = width_points[-2], width_points[-1]
            canvas.create_line(x_pre, y_pre, x, y, fill="green", tags="tmp")
    else:
        return


def end_draw(event):
    """
    按 esc 键结束当前绘制
    """
    if draw_tag == 0:
        if len(lanes_points) == 0 or len(lanes_points) == 1:
            return
        lane_id = simpledialog.askstring(
            title="获取信息", prompt="输入车道ID", initialvalue="000"
        )
        while lane_id in lanes:
            lane_id += "_"
        lanes[lane_id] = lanes_points.copy()
        lanes_points.clear()
    elif draw_tag == 1:
        if len(rois_points) == 0 or len(rois_points) == 1:
            return
        roi_id = simpledialog.askstring(
            title="获取信息", prompt="输入ROI_ID", initialvalue="r00"
        )
        while roi_id in rois:
            roi_id += "_"
        rois[roi_id] = rois_points.copy()
        rois_points.clear()
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
ww = 1500
wh = 900
x = (sw - ww) / 2
y = (sh - wh) / 2
main.title("检测区域配置")
main.geometry("%dx%d+%d+%d" % (ww, wh, x, y))
main.resizable(0, 0)
main.bind_all("<Escape>", end_draw)

# 创建左侧toolbar
toolbar = tk.Frame(main, borderwidth=1, relief=tk.RIDGE)
toolbar.pack(side=tk.LEFT, fill=tk.Y)

btn_openimg = tk.Button(
    toolbar, text="打开图片", height=3, width=10, command=open_image
)
btn_drawpoly1 = tk.Button(toolbar, text="车道", height=3, width=10, command=drawpoly1)
btn_drawpoly2 = tk.Button(toolbar, text="ROI", height=3, width=10, command=drawpoly2)
btn_drawpoly3 = tk.Button(toolbar, text="宽度", height=3, width=10, command=drawpoly3)
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

roi_info_box = tk.Listbox(info_box, width=50)
roi_info_box.pack(side=tk.RIGHT, fill=tk.Y)
roi_info_box.bind("<Button-1>", choose_polygon1)
roi_info_box.bind("<Button-3>", choose_polygon_delete1)

main.mainloop()
