<<<<<<< HEAD
import os
import pandas as pd

# 替换为您本地的文件夹路径
folder_path = r'Y:\Desktop\temp'

# 替换为您希望保存Excel文件的实际文件路径
output_file_path = r'Y:\Desktop\temp\合并后的文件.xlsx'

# 获取文件夹中所有的Excel文件
excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]

all_sheets = []

# 遍历每个文件并读取每个工作表
for file_path in excel_files:
    with pd.ExcelFile(file_path) as xls:
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            all_sheets.append(df)

# 合并所有工作表
combined_df = pd.concat(all_sheets, ignore_index=True)

# 删除具有相同'业务流水号'的行
combined_df.drop_duplicates(subset='业务流水号', keep='first', inplace=True)

# 将'交易时间'列转换为datetime类型，以便正确排序
combined_df['交易时间'] = pd.to_datetime(combined_df['交易时间'])

# 按'交易时间'排序
combined_df.sort_values(by='交易时间', ascending=True, inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# 查看合并和排序后的DataFrame
print(combined_df)

# 保存最终的DataFrame为Excel文件
combined_df.to_excel(output_file_path, index=False, sheet_name='合并后的数据')
=======
import json
import os
import pathlib
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageAnnotationTool:
    def __init__(self):
        # Initialize main window
        self.main = tk.Tk()
        self.main.title("图片绘制工具")
        self.set_window_geometry()
        self.main.resizable(0, 0)
        self.main.bind_all("<Escape>", self.end_draw)

        # Define global variables
        self.skip_seconds = 1  # Amount of seconds to skip when reading from a video
        self.flag_draw = False  # Flag to check if we can draw or not
        self.current_image = ""  # Path to the current image
        self.images = []  # List to store the current image
        self.polygons = []  # List to store drawn polygons
        self.points = []  # List to store points of the current polygon being drawn
        self.config_dir = None  # Directory where the config file is located

        # GUI Components
        self.create_toolbar()
        self.create_canvas()
        self.create_info_box()

    def set_window_geometry(self):
        sw, sh = self.main.winfo_screenwidth(), self.main.winfo_screenheight()
        ww, wh = 1920, 1080
        x, y = (sw - ww) / 2, (sh - wh) / 2
        self.main.geometry("%dx%d+%d+%d" % (ww, wh, x, y))

    def create_toolbar(self):
        """Create the toolbar with buttons for various operations."""
        self.toolbar = tk.Frame(self.main, borderwidth=1, relief=tk.RIDGE)
        self.toolbar.pack(side=tk.LEFT, fill=tk.Y)

        # Define and pack buttons for the toolbar
        btn_opendir = tk.Button(
            self.toolbar, text="打开文件夹", height=3, width=10, command=self.open_directory
        )
        btn_configdir = tk.Button(
            self.toolbar,
            text="配置文件夹",
            height=3,
            width=10,
            command=self.open_config_directory,
        )
        btn_openimage = tk.Button(
            self.toolbar, text="打开图片", height=3, width=10, command=self.open_one_image
        )
        btn_opendir.pack()
        btn_configdir.pack()
        btn_openimage.pack()

    def create_canvas(self):
        """Create the canvas for image display and annotation."""
        self.canvas = tk.Canvas(self.main, borderwidth=1, relief=tk.RIDGE)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.draw_point)
        self.canvas.bind("<Motion>", self.draw_reference_line)

    def create_info_box(self):
        """Create the info box for displaying image and polygon info."""
        self.info_box = tk.Frame(self.main, height=200)
        self.info_box.pack(side=tk.BOTTOM, fill=tk.X)

        # File list
        self.file_list = tk.Listbox(self.info_box, yscrollcommand=tk.TRUE)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_list.bind("<Double-Button-1>", self.open_image)

        # Polygon info list
        self.video_info_box = tk.Listbox(self.info_box, width=30)
        self.video_info_box.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_info_box.bind("<Button-1>", self.choose_polygon)
        self.video_info_box.bind("<Button-3>", self.choose_polygon_delete)

    def end_draw(self, event):
        """End the current drawing of a polygon."""
        if len(self.points) <= 1:
            return
        self.polygons.append(self.points.copy())
        self.points.clear()
        self.refresh_canvas()
        self.refresh_info_box()

    def draw_point(self, event):
        """Draw a single point in a polygon."""
        # The rest of the function (similar to original but with references to 'self.')

    def draw_reference_line(self, event):
        """Draw a reference line to guide drawing of polygons."""
        # The rest of the function (similar to original but with references to 'self.')

    def open_directory(self):
        """Open a directory and list all images in it."""
        # The rest of the function (similar to original but with references to 'self.')

    def open_config_directory(self):
        """Set the directory where the config file is located."""
        self.config_dir = filedialog.askdirectory()

    def open_one_image(self):
        """Open a single image file."""
        # The rest of the function (similar to original but with references to 'self.')

    def refresh_canvas(self):
        """Refresh the canvas to display the current image and polygons."""

        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.delete(tk.ALL)

        for image in self.images:
            self.canvas.create_image(image[0], image[1], anchor=tk.NW, image=image[2])

        for polygon in self.polygons:
            self.canvas.create_polygon(polygon, outline="red", fill="")

        for idx in range(len(self.points) // 2 - 1):
            sx = self.points[idx * 2]
            sy = self.points[idx * 2 + 1]
            ex = self.points[idx * 2 + 2]
            ey = self.points[idx * 2 + 3]
            self.canvas.create_line(sx, sy, ex, ey, fill="red")

    def refresh_info_box(self):
        """Refresh the info box with the current image and polygon info."""
        # The rest of the function (similar to original but with references to 'self.')
        self.video_info_box.delete(0, self.video_info_box.size())
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y, w, h = (
            self.images[0][0],
            self.images[0][1],
            self.images[0][2].width(),
            self.images[0][2].height(),
        )
        for rect in self.polygons:
            self.video_info_box.insert(
                tk.END,
                "{:0>3d}-{:0>3d}-{:0>3d}-{:0>3d}".format(
                    int(rect[0] - x),
                    int(rect[1] - y),
                    int(rect[2] - x),
                    int(rect[3] - y),
                ),
            )

    def choose_polygon(self, event):
        """Select a specific polygon."""
        # The rest of the function (similar to original but with references to 'self.')

    def choose_polygon_delete(self, event):
        """Delete a selected polygon."""
        # The rest of the function (similar to original but with references to 'self.')

    # You can add other methods (for opening images, saving, etc.) similarly.


# Running the tool
if __name__ == "__main__":
    tool = ImageAnnotationTool()
    tool.main.mainloop()
>>>>>>> refs/remotes/origin/main
