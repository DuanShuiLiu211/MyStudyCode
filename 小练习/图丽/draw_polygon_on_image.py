import json
import os
import pathlib
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk


main = tk.Tk()


sw = main.winfo_screenwidth()
sh = main.winfo_screenheight()

# 窗口长宽
ww = 1920
wh = 1080

x = (sw - ww) / 2
y = (sh - wh) / 2

global end_draw


def end_draw(event):
    """
    按 esc 键结束当前绘制
    """
    if len(points) == 0 or len(points) == 1:
        return
    polygons.append(points.copy())
    points.clear()
    refresh()
    refresh_info()


main.title("图片绘制工具")
main.geometry('%dx%d+%d+%d' % (ww, wh, x, y))
main.resizable(0, 0)
main.bind_all('<Escape>', end_draw)

toolbar = tk.Frame(main, borderwidth=1, relief=tk.RIDGE)
toolbar.pack(side=tk.LEFT, fill=tk.Y)

# 从视频中读取图片的时候跳过的秒数，视频的第一帧可能模糊
skip_seconds = 1
# 标记当前能够进行绘制，目前没有使用
flag_draw = False
# 当前读取的视频的路径
current_image = ''
# 当前读取的视频的帧
images = []
# 图片中绘制的多边形
polygons = []
# 当前绘制的多边形的点集合
points = []
# 配置文件所在的文件夹
config_dir = None   


def draw_point(event):
    """
    绘制多边形中的单个点
    """
    x, y = event.x, event.y
    if len(images) == 0:
        return
    img_x, img_y = images[0][0], images[0][1], 
    img_w, img_h = images[0][2].width(), images[0][2].height()
    if x < img_x or x > img_x + img_w or y < img_y or y > img_y + img_h:
        return
    points.extend([x, y])


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

    if len(points) == 0:
        return
    x_pre, y_pre = points[-2], points[-1]

    canvas.create_line(x_pre, y_pre, x, y, fill='red', tags="tmp")
    # points_ = []
    # for point in points:
    #     x = point['x']
    #     y = point['y']
    #     points_.extend([x, y])
    # print(points)
    # canvas.create_polygon(points, outline='red', fill='red', tags="tmp")
    # canvas.create_rectangle(x_pre, y_pre, x, y, tags="tmp")


def rect_click(event):
    # print('--')
    # print(event)
    pass


def refresh():
    """
    刷新画布
    """
    w, h = canvas.winfo_width(), canvas.winfo_height()
    canvas.delete(tk.ALL)

    for image in images:
        canvas.create_image(image[0], image[1], anchor=tk.NW, image=image[2])

    for polygon in polygons:
        canvas.create_polygon(polygon, outline='red', fill='')
        # random_num = random.random()
        # rect_obj = canvas.create_rectangle(rect[0], rect[1], rect[2], rect[3], tags="tags")
        # polygon_obj.append(rect_obj)

        canvas.tag_bind("tags", '<Button-1>', rect_click)
        canvas.tag_bind("tags", '<Button-2>', rect_click)
        canvas.tag_bind("tags", '<Button-3>', rect_click)

    for idx in range(len(points) // 2 - 1):
        sx = points[idx * 2]
        sy = points[idx * 2 + 1]
        ex = points[idx * 2 + 2]
        ey = points[idx * 2 + 3]
        canvas.create_line(sx, sy, ex, ey, fill='red')
    # if len(points) == 0:
    #     return
    # canvas.create_polygon(points, outline='red', fill='')


def refresh_info():
    """
    刷新当前视频的标记框信息
    """
    video_info_box.delete(0, video_info_box.size())
    canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
    x, y, w, h = images[0][0], images[0][1], images[0][2].width(
    ), images[0][2].height()
    for rect in polygons:
        video_info_box.insert(tk.END, '{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}'.format(
            int(rect[0] - x), int(rect[1] - y), int(rect[2] - x), int(rect[3] - y)))


canvas = tk.Canvas(main, borderwidth=1, relief=tk.RIDGE)
canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas.bind('<Button-1>', draw_point)
canvas.bind('<Motion>', draw_reference_line)


info_box = tk.Frame(main, height=200)
info_box.pack(side=tk.BOTTOM, fill=tk.X)


def open_image(event, image_path=None):
    """
    打开视频，读取视频的某一帧并显示到画布中，同时读取当前图片对应的标记信息显示在信息框中
    同时将标记信息中记录的多边形绘制到画布中
    """
    if image_path is None:
        selection = file_list.curselection()
        if selection is None or len(selection) == 0:
            return
        file_path = file_list.get(selection)
    else:
        file_path = image_path
    global current_image
    current_image = file_path
    polygons.clear()

    canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
    frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags=cv2.IMREAD_COLOR)

    x, y = 0, 0
    image_w, image_h = frame.shape[1], frame.shape[0]
    if (float(image_w / image_h) > float(canvas_w / canvas_h)):
        frame = cv2.resize(
            frame, (canvas_w, int(canvas_w * image_h / image_w)))
        x = 0
        y = (canvas_h - frame.shape[0]) // 2
    else:
        frame = cv2.resize(
            frame, (int(canvas_h * image_w / image_h), canvas_h))
        x = (canvas_w - frame.shape[1]) // 2
        y = 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=frame)

    image = (x, y, img, image_w, image_h)
    images.clear()
    images.append(image)

    # 读取配置文件
    if config_dir is None:
        config_path = os.path.splitext(file_path)[0] + '.json'
    else:
        config_path = os.path.join(config_dir, os.path.splitext(os.path.split(file_path)[1])[0] + '.json')

        
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

            for region in config['regions']:
                points = region['ps']
                points_ = []
                for point in points:
                    x_ = x + img.width() * int(point['x']) / image_w
                    y_ = y + img.height() * int(point['y']) / image_h
                    points_.extend([x_, y_])

                polygons.append(points_)

    refresh()
    refresh_info()



file_list = tk.Listbox(info_box, yscrollcommand=tk.TRUE)
file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
file_list.bind('<Double-Button-1>', open_image)


def choose_polygon(event):
    # 高亮选择的多边形框
    selection = video_info_box.curselection()
    if len(selection) == 0:
        return
    polygon = polygons[selection[0]]
    canvas.delete('tmp2')
    canvas.create_polygon(polygon, outline='red', fill='red', tags="tmp2")


def choose_polygon_delete(event):
    """
    选中当前打开的视频的标记框，用于删除
    """
    def delete():
        selection = video_info_box.curselection()
        if len(selection) == 0:
            return
        positions = video_info_box.get(selection).split("-")
        positions = [int(pos) for pos in positions]
        canvas.delete('tmp2')
        polygons.remove(polygons[selection[0]])
        video_info_box.delete(selection)
        refresh()

    video_info_box.send
    menu = tk.Menu(main, tearoff=0)
    menu.add_command(label="删除", command=delete)
    menu.post(event.x_root, event.y_root)


video_info_box = tk.Listbox(info_box, width=30)
video_info_box.pack(side=tk.RIGHT, fill=tk.Y)
video_info_box.bind('<Button-1>', choose_polygon)
video_info_box.bind('<Button-3>', choose_polygon_delete)


def draw():
    """
    开始绘制
    """
    flag_draw = True


def opendir():
    """
    读取选中的文件夹中的全部视频，支持 mp4，avi，mkv 等格式
    """
    dir_path = filedialog.askdirectory()
    if dir_path is None: return
    path = pathlib.Path(dir_path)

    images = path.glob("*.[jp][pn][gg]")
    file_list.delete(0, tk.END)
    for video in images:
        # print(video)
        file_list.insert(tk.END, video)


def open_configdir():
    """
    打开配置文件的文件夹
    """
    global config_dir
    config_dir = filedialog.askdirectory()


def open_one_image():
    """
    读取单个视频文件
    """
    file_path = filedialog.askopenfile()
    if file_path is None:
        return
    file_path = file_path.name

    ext = os.path.splitext(file_path)[-1]
    if ext not in ['.jpg', '.jpeg', '.png']: return
    file_list.delete(0, tk.END)
    file_list.insert(tk.END, file_path)


def pre_image():
    """
    打开当前打开的视频的前一个视频
    """
    index = file_list.index('active')
    if index - 1 < 0:
        return
    else:
        file_path = file_list.get(index - 1)
        file_list.activate(index - 1)
        file_list.see(index - 1)
        open_image(None, image_path=file_path)


def next_image():
    """
    打开当前打开的视频的后一个视频
    """
    index = file_list.index('active')
    # print(index, file_list.size())
    if index + 1 >= file_list.size():
        return
    else:
        file_path = file_list.get(index + 1)
        file_list.activate(index + 1)
        file_list.see(index - 1)
        open_image(None, image_path=file_path)


def save():
    """
    保存配置文件
    """
    if len(current_image) == 0:
        return
    file_path = current_image
    x, y, w, h, w_origin, h_origin = images[0][0], images[0][1], images[0][2].width(
    ), images[0][2].height(), images[0][3], images[0][4]

    result = {}
    regions = []
    for polygon in polygons:
        points_ = []
        for idx in range(len(polygon) // 2):
            x_ = int((polygon[idx * 2] - x) / w * w_origin)
            y_ = int((polygon[idx * 2 + 1] - y) / h * h_origin)
            points_.append({'x': x_, 'y': y_})
        regions.append({'ps': points_})

    result['regions'] = regions

    
    if config_dir is None:
        config_path = os.path.splitext(file_path)[0] + '.json'
    else:
        config_path = os.path.join(config_dir, os.path.splitext(os.path.split(file_path)[1])[0] + '.json')
    print("images path: {}".format(file_path))
    print("config path: {}".format(config_path))
    with open(config_path, 'w') as f:
        json.dump(result, f)


btn_opendir = tk.Button(toolbar, text="打开文件夹", height=3,
                        width=10, command=opendir)
btn_configdir = tk.Button(toolbar, text="配置文件夹", height=3,
                        width=10, command=open_configdir)
btn_openvideo = tk.Button(toolbar, text="打开图片", height=3,
                          width=10, command=open_one_image)
btn_drawrect = tk.Button(toolbar, text="绘制矩形",
                         height=3, width=10, command=draw)
btn_pre_image = tk.Button(toolbar, text="上一个", height=3,
                          width=10, command=pre_image)
btn_next_image = tk.Button(
    toolbar, text="下一个", height=3, width=10, command=next_image)
btn_save = tk.Button(toolbar, text="保存", height=3, width=10, command=save)

btn_opendir.pack(side=tk.TOP)
btn_configdir.pack(side=tk.TOP)
btn_openvideo.pack(side=tk.TOP)
btn_drawrect.pack(side=tk.TOP)
btn_pre_image.pack(side=tk.TOP)
btn_next_image.pack(side=tk.TOP)
btn_save.pack(side=tk.TOP)

main.mainloop()
