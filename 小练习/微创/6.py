import os
import shutil
from glob import glob

import matplotlib.pyplot as plt


def add_prefix(i):
    """
    args: i:a number
          num_prefix: total number of the word
        return: added number
    """
    i_temp = str(i)
    if len(i_temp) == 1:
        return "000" + i_temp
    elif len(i_temp) == 2:
        return "00" + i_temp
    elif len(i_temp) == 3:
        return "0" + i_temp


pngpath = "/home/gyc/DL_learning/microport/视频分割-final/第三批/*.png"
pnglist = glob(pngpath)

targetPath = "/home/gyc/DL_learning/microport/视频分割-final/第三批颈总动脉视频截取/"

foldlist = (
    set(glob("/home/gyc/DL_learning/microport/视频分割-final/第三批/*"))
    - set(glob("/home/gyc/DL_learning/microport/视频分割-final/第三批/*.png"))
    - set(glob("/home/gyc/DL_learning/microport/视频分割-final/第三批/*.avi"))
)
foldlist = list(foldlist)

for i in range(len(foldlist)):
    selectedframe = glob(foldlist[i] + "*.png")
    frames = []
    for j in range(len(selectedframe)):
        frames.append(selectedframe[j].split("_")[-1].replace(".png", ""))
    if len(frames) > 0:
        start_frame = (
            int(frames[0]) - 1 if int(frames[1]) > int(frames[0]) else int(frames[1])
        )
        end_frame = (
            int(frames[0]) - 1 if int(frames[0]) > int(frames[1]) else int(frames[1])
        )
        imgList = glob(foldlist[i] + "/*.png")
        imgList.sort()
        if not os.path.exists(targetPath + foldlist[i].split("/")[-1]):
            os.mkdir(targetPath + foldlist[i].split("/")[-1])
        for k in range(len(imgList)):
            if (int(imgList[k].split("/")[-1].replace(".png", "")) >= start_frame) & (
                int(imgList[k].split("/")[-1].replace(".png", "")) <= end_frame
            ):
                shutil.copy(imgList[k], imgList[k].replace("第三批", "第三批颈总动脉视频截取"))
