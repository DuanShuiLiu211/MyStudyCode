import cv2
import os
import json
import tkinter as tk
import numpy as np
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
from collections import OrderedDict
import tkinter.simpledialog as sd

image = None
attr_polygons = list()
attr_points = list()
out_polygons = list()
out_points = list()
config_path = None
draw_tag = 0  # 绘制类型 0：多边形, 1：矩形

lanes = OrderedDict()
lanes_points = []
WIDTH = 3


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
    if image is None:
        return
    canvas.create_image(image[0], image[1], anchor=tk.NW, image=image[2])

    for polygon in attr_polygons:
        canvas.create_polygon(polygon, outline='red', fill='', width=WIDTH)
        canvas.tag_bind("tags", '<Button-1>', rect_click)
        canvas.tag_bind("tags", '<Button-2>', rect_click)
        canvas.tag_bind("tags", '<Button-3>', rect_click)
    for idx in range(len(attr_points) // 2 - 1):
        sx = attr_points[idx * 2]
        sy = attr_points[idx * 2 + 1]
        ex = attr_points[idx * 2 + 2]
        ey = attr_points[idx * 2 + 3]
        canvas.create_line(sx, sy, ex, ey, fill='red', width=WIDTH)

    for polygon in out_polygons:
        print("polygon:\n", polygon)
        canvas.create_rectangle(polygon, outline='red', fill='', width=WIDTH)
        canvas.tag_bind("tags", '<Button-1>', rect_click)
        canvas.tag_bind("tags", '<Button-2>', rect_click)
        canvas.tag_bind("tags", '<Button-3>', rect_click)
    for idx in range(len(out_points) // 2 - 1):
        sx = out_points[idx * 2]
        sy = out_points[idx * 2 + 1]
        ex = out_points[idx * 2 + 2]
        ey = out_points[idx * 2 + 3]
        print("sx:{} sy:{} ex:{} ey:{}:\n".format(sx, sy, ex, ey))
        canvas.create_rectangle(sx,
                                sy,
                                ex,
                                ey,
                                fill='',
                                outline='red',
                                width=WIDTH)
        # out_points.clear()


def refresh_info():
    """
    刷新当前标记框信息
    """
    info_box.delete(0, info_box.size())
    canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
    x, y, w, h = image[0], image[1], image[2].width(), image[2].height()
    print("x:{} and y:{} in refresh_info:\n".format(x, y))

    #多边形
    for polyon in attr_polygons:
        points = polyon.copy()
        print("points in refresh_info:\n", points)
        #canvas上的坐标
        pointStr = ''
        for j in range(len(points)):
            pointStr += str(points[j])
            if j != len(points) - 1:
                pointStr += "-"

        #原图上的坐标(图像没有缩放前的坐标)
        if y == 0:
            for i in range(len(points)):
                if i % 2 == 0:
                    points[i] = int(
                        (points[i] - x) *
                        (image_w / int(canvas_h * image_w / image_h)))
                else:
                    points[i] = int((points[i] - y) * (image_h / canvas_h))
        elif x == 0:
            for i in range(len(points)):
                if i % 2 == 0:
                    points[i] = int((points[i] - x) * (image_w / canvas_w))
                else:
                    points[i] = int(
                        (points[i] - y) *
                        (image_h / int(canvas_w * image_h / image_w)))
        pointStr = pointStr + ':'
        for j in range(len(points)):
            pointStr += str(points[j])
            if j != len(points) - 1:
                pointStr += "-"
        info_box.insert(tk.END, pointStr)
        # info_box.insert(tk.END, '{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}'.format(
        #     int(rect[0] - x), int(rect[1] - y), int(rect[2] - x), int(rect[3] - y)))

    #矩形
    for rect in out_polygons:
        points = rect.copy()
        print("points in refresh_info:\n", points)
        #canvas上的坐标
        pointStr = ''
        for j in range(len(points)):
            pointStr += str(points[j])
            if j != len(points) - 1:
                pointStr += "-"

        #原图上的坐标(图像没有缩放前的坐标)
        if y == 0:
            for i in range(len(points)):
                if i % 2 == 0:
                    points[i] = int(
                        (points[i] - x) *
                        (image_w / int(canvas_h * image_w / image_h)))
                else:
                    points[i] = int((points[i] - y) * (image_h / canvas_h))
        elif x == 0:
            for i in range(len(points)):
                if i % 2 == 0:
                    points[i] = int((points[i] - x) * (image_w / canvas_w))
                else:
                    points[i] = int(
                        (points[i] - y) *
                        (image_h / int(canvas_w * image_h / image_w)))
        pointStr = pointStr + ':'
        for j in range(len(points)):
            pointStr += str(points[j])
            if j != len(points) - 1:
                pointStr += "-"

        info_box.insert(tk.END, pointStr)
        # info_box.insert(tk.END, '{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}'.format(
        #     int(rect[0] - x), int(rect[1] - y), int(rect[2] - x), int(rect[3] - y)))


def open_image():
    """
    读取图片，同时加载已有检测区域配置
    """
    file_path = filedialog.askopenfile(
        filetypes=[('Image Files',
                    ['*.jpg', '*.jpeg', '*.png']), ('All Files', '*')])
    if file_path is None:
        return
    file_path = file_path.name
    global canvas_w, canvas_h
    canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
    frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                         flags=cv2.IMREAD_COLOR)

    x, y = 0, 0
    global image_w, image_h
    image_w, image_h = frame.shape[1], frame.shape[0]
    print("canvas_w-{} canvas_h-{} image_w-{} image_h-{}".format(
        canvas_w, canvas_h, image_w, image_h))
    if (float(image_w / image_h) > float(canvas_w / canvas_h)):
        frame = cv2.resize(frame,
                           (canvas_w, int(canvas_w * image_h / image_w)))
        print("the shape of frame:w-{} h-{}:\n".format(frame.shape[1],
                                                       frame.shape[0]))
        x = 0
        y = (canvas_h - frame.shape[0]) // 2
    else:
        frame = cv2.resize(frame,
                           (int(canvas_h * image_w / image_h), canvas_h))
        print("the shape of frame:w-{} h-{}:\n".format(frame.shape[1],
                                                       frame.shape[0]))
        x = (canvas_w - frame.shape[1]) // 2
        y = 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=frame)
    global image
    image = (x, y, img, image_w, image_h)


def choose_polygon(event):
    # 高亮选择的多边形框
    selection = info_box.curselection()
    if len(selection) == 0:
        return
    polygons = attr_polygons + out_polygons
    polygon = polygons[selection[0]]
    canvas.delete('tmp2')
    if polygon in attr_polygons:
        canvas.create_polygon(polygon, outline='red', fill='red', tags="tmp2")
    else:
        canvas.create_rectangle(polygon,
                                outline='blue',
                                fill='blue',
                                tags="tmp2")


def choose_polygon_delete(event):
    """
    选中当前打开的视频的标记框，用于删除
    """

    def delete():
        selection = info_box.curselection()
        if len(selection) == 0:
            return
        # positions = info_box.get(selection).split("-")
        positions1 = info_box.get(selection).split(":")[0]
        positions2 = info_box.get(selection).split(":")[1]
        positions = positions1 + "_" + positions2
        positions = positions.split("-")
        positions = [int(pos) for pos in positions]
        canvas.delete('tmp2')
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


def draw_point(event):
    """
    绘制多边形中的单个点
    """
    x, y = event.x, event.y
    print("x-%d y-%d" % (x, y))
    if image is None:
        return
    img_x, img_y = image[0], image[1],
    img_w, img_h = image[2].width(), image[2].height()
    if x < img_x or x > img_x + img_w or y < img_y or y > img_y + img_h:
        return
    if draw_tag == 0:
        attr_points.extend([x, y])
    elif draw_tag == 1:
        out_points.extend([x, y])
        print("out_points:\n", out_points)
    elif draw_tag == 2:
        lanes_points.extend([x, y])
    else:
        return


def draw_reference_shape(event):
    """
    移动鼠标，绘制参考形状
    """
    # 清除画布并绘制保存的内容
    # print("result in reference_shape:\n",result,len(result))
    if result is not None and len(result) == 0:
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
            canvas.create_line(x_pre,
                               y_pre,
                               x,
                               y,
                               fill='red',
                               tags="tmp",
                               width=WIDTH)
        elif draw_tag == 1:
            #绘制矩形
            if len(out_points) == 0:
                return
            x_pre, y_pre = out_points[0], out_points[1]
            print("x_pre:{} y_pre:{} x:{} y:{}:\n".format(x_pre, y_pre, x, y))
            canvas.create_rectangle(x_pre,
                                    y_pre,
                                    x,
                                    y,
                                    outline="red",
                                    tags="tmp",
                                    width=WIDTH)

        elif draw_tag == 2:
            if len(lanes_points) == 0:
                return
            x_pre, y_pre = lanes_points[-2], lanes_points[-1]
            canvas.create_line(x_pre, y_pre, x, y, fill='green', tags="tmp")
        else:
            return
    else:
        pass


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
        lane_id = simpledialog.askstring(title='获取信息',
                                         prompt='输入车道ID',
                                         initialvalue='000')
        while lane_id in lanes:
            lane_id += '_'
        lanes[lane_id] = lanes_points.copy()
        lanes_points.clear()
    refresh()
    refresh_info()


def drawpoly():
    global draw_tag
    draw_tag = 0


def drawrect():
    global draw_tag
    draw_tag = 1


def dialog(event):
    # 使用simpledialog.askstring()方法获取用户输入的字符串
    x, y, w, h = image[0], image[1], image[2].width(), image[2].height()
    global result
    result = sd.askstring("Input", "Enter some text:")
    print("result:\n", result)
    # 将用户输入的字符串插入到Text小部件中
    if result is not None:
        # text_box.insert("2.0", result)
        if image is None:
            return
        #增加功能：绘制多个shape
        # resultList = result.split(',')
        resultList = result.split('.')
        print("resultList:\n", resultList)
        for shapeIndex in range(len(resultList)):
            if shapeIndex != len(resultList) - 1:
                shapecor = resultList[shapeIndex]
                print("shapecor:\n", shapecor)
                shapeCorList = shapecor.split(',')
                #图像坐标
                for index in range(len(shapeCorList)):
                    if index % 2 == 0:
                        # shapeCorList[index] = int(shapeCorList[index])+x
                        shapeCorList[index] = int(shapeCorList[index])
                        pass
                    else:
                        # shapeCorList[index] = int(shapeCorList[index])+y
                        shapeCorList[index] = int(shapeCorList[index])
                        pass
                #图像坐标转canvas坐标
                import copy
                canvasShapeCorList = copy.deepcopy(shapeCorList)
                if y == 0:
                    for i in range(len(canvasShapeCorList)):
                        if i % 2 == 0:
                            canvasShapeCorList[i] = int(
                                canvasShapeCorList[i] /
                                (image_w / int(canvas_h * image_w / image_h)) +
                                x)
                        else:
                            canvasShapeCorList[i] = int(canvasShapeCorList[i] /
                                                        (image_h / canvas_h) +
                                                        y)
                elif x == 0:
                    for i in range(len(canvasShapeCorList)):
                        if i % 2 == 0:
                            canvasShapeCorList[i] = int(canvasShapeCorList[i] /
                                                        (image_w / canvas_w) +
                                                        x)
                        else:
                            canvasShapeCorList[i] = int(
                                canvasShapeCorList[i] /
                                (image_h / int(canvas_w * image_h / image_w)) +
                                y)
                cordinate = result + str(canvasShapeCorList)
                text_box.insert("2.0", cordinate)

                if len(shapeCorList) == 4:
                    canvas.create_rectangle(canvasShapeCorList,
                                            outline='red',
                                            fill='',
                                            tags='cor',
                                            width=WIDTH)
                elif len(shapeCorList) > 4:
                    canvas.create_polygon(canvasShapeCorList,
                                          outline='red',
                                          fill='',
                                          tags='cor',
                                          width=WIDTH)
                else:
                    print("输入坐标的格式有误，请重新输入！")
                    return


def delete(event):
    text_box.delete("2.0", "end")
    canvas.delete('cor')
    global result
    result = ""


# GUI部分
main = tk.Tk()
sw = main.winfo_screenwidth()
sh = main.winfo_screenheight()
ww = 1920
wh = 1080
x = (sw - ww) / 2
y = (sh - wh) / 2
main.title("图片辅助标注工具")
main.geometry('%dx%d+%d+%d' % (ww, wh, x, y))
main.resizable(0, 0)
main.bind_all('<Escape>', end_draw)

# 创建左侧toolbar
toolbar = tk.Frame(main, borderwidth=1, relief=tk.RIDGE)
toolbar.pack(side=tk.LEFT, fill=tk.Y)

btn_openimg = tk.Button(toolbar,
                        text="打开图片",
                        height=3,
                        width=10,
                        command=open_image)
btn_drawpoly = tk.Button(toolbar,
                         text="polygon",
                         height=3,
                         width=10,
                         command=drawpoly)
btn_rectangle = tk.Button(toolbar,
                          text="rectangle",
                          height=3,
                          width=10,
                          command=drawrect)

btn_openimg.pack(side=tk.TOP)
btn_drawpoly.pack(side=tk.TOP)
btn_rectangle.pack(side=tk.TOP)
# btn_drawpoly2.pack(side=tk.TOP)
# btn_drawpoly3.pack(side=tk.TOP)
# btn_save.pack(side=tk.TOP)

# 右侧的canvas
canvas = tk.Canvas(main, borderwidth=1, relief=tk.RIDGE)
canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

canvas.bind('<Button-1>', draw_point)  # 鼠标左键点击绘制点
canvas.bind('<Motion>', draw_reference_shape)  # 鼠标拖动绘制线段

# 底部的infobox
info_box = tk.Listbox(main, height=10)
info_box.pack(side=tk.BOTTOM, fill=tk.X)
info_box.bind('<Button-1>', choose_polygon)  # 鼠标左键高亮多边形区域
info_box.bind('<Button-3>', choose_polygon_delete)  # 鼠标右键出现删除选项

#create text box
text_box = tk.Text(info_box, width=120)
text_box.pack(side=tk.RIGHT, fill=tk.Y)
text_box.insert("2.0", "请输入坐标，坐标格式为x1,y1,x2,y2,...:\n")
result = ""

text_box.bind('<Button-1>', dialog)
text_box.bind('<Button-3>', delete)  # 鼠标右键出现删除选项

main.mainloop()