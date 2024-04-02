import cv2
import numpy as np


def detect_face_cascade():
    # 使用Haar级联检测器进行实时人脸检测
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取摄像头图像
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在灰度图上检测人脸
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # 在检测到的人脸周围绘制矩形框
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示结果图像
        cv2.imshow("Face Detection", frame)

        # 按下ESC键退出
        if cv2.waitKey(1) == 27:
            break

    # 关闭摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()


def detect_face_net():
    # 使用DNN模型进行实时人脸检测
    # 加载预训练模型
    net = cv2.dnn.readNetFromCaffe(
        "assets/opencv_dnn/deploy.prototxt",
        "assets/opencv_dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    )

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 对图像进行预处理
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )

        # 输入预处理后的图像到网络中
        net.setInput(blob)
        detections = net.forward()

        # 解析检测结果
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 设置置信度阈值
                box = detections[0, 0, i, 3:7] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                )
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("Real-time Face Detection", frame)

        # 按下ESC键退出
        if cv2.waitKey(1) == 27:
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_face_cascade()
    # detect_face_net()
