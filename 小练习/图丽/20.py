import cv2
import numpy as np

model = cv2.dnn.readNetFromCaffe(
    "./assets/opencv_dnn/vehicle_classify/deploy.prototxt",
    "./assets/opencv_dnn/vehicle_classify/model.caffemodel",
)

img_path = "./assets/data/imgs/3.png"
img_raw = cv2.resize(cv2.imread(img_path), (224, 224))

mean_rgb = np.array([123.68, 116.779, 103.939], dtype=np.float32)
std_rgb = np.array([58.393, 57.12, 57.375], dtype=np.float32)

img_normalized = (img_raw.astype(np.float32) - mean_rgb) / std_rgb
img_input = img_normalized.transpose((2, 0, 1))[np.newaxis, ::-1, ...]

label_list = open("./assets/opencv_dnn/vehicle_classify/label.txt", "r").readlines()
model.setInput(img_input)
embedding = model.forward()[0]

score = embedding.transpose().max()
idx = embedding.transpose().argmax()
label = label_list[idx]

print(f"score: {score}, idx: {idx}, label: {label}")
