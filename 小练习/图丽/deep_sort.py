import os
import cv2
import torch
import datetime
import numpy as np
import logging
from collections import defaultdict
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义命令行标志
flags.DEFINE_string("model_path", "./models/yolov10.pt", "检测模型的权重文件路径")
flags.DEFINE_string("classes_path", "./configs/coco.txt", "检测模型的类别名称文件路径")
flags.DEFINE_string(
    "inputs", "./datas/test_1.mp4", "输入视频文件路径或摄像头索引（如0）"
)
flags.DEFINE_string("outputs", "./results/output.mp4", "保存结果的视频文件路径")
flags.DEFINE_float("confidence_threshold", 0.50, "检测置信度阈值")
flags.DEFINE_integer("class_blur_id", None, "需要应用高斯模糊的类别ID")
flags.DEFINE_integer("class_id", None, "需要跟踪的类别ID（若未指定，则跟踪所有类别）")

FLAGS = flags.FLAGS


def initialize_video_capture(video_input):
    """初始化视频捕获设备（文件或摄像头）。"""
    if video_input.isdigit():
        video_input = int(video_input)
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        logger.error("错误：无法打开视频源。")
        raise ValueError("无法打开视频源")
    return cap


def initialize_model(model_path):
    """从指定路径加载模型并设置适当的设备（CPU/GPU）。"""
    if not os.path.exists(model_path):
        logger.error(f"在 {model_path} 未找到模型权重文件")
        raise FileNotFoundError("未找到模型权重文件")

    model = torch.load(model_path)
    model.eval()  # 设置模型为评估模式

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    logger.info(f"使用 {device} 作为处理设备")
    return model


def load_class_names(classes_path):
    """从指定文件加载类别名称。"""
    if not os.path.exists(classes_path):
        logger.error(f"在 {classes_path} 未找到类别名称文件")
        raise FileNotFoundError("未找到类别名称文件")

    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names


def process_frame(frame, model, tracker, class_names, colors):
    """处理单帧以进行对象检测和跟踪。"""
    # 执行对象检测
    results = model(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if FLAGS.class_id is None:
            if confidence < FLAGS.confidence_threshold:
                continue
        else:
            if class_id != FLAGS.class_id or confidence < FLAGS.confidence_threshold:
                continue

        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # 更新跟踪器
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks


def draw_tracks(
    frame, tracks, class_names, colors, class_counters, track_class_mapping
):
    """在帧上绘制跟踪结果，并根据需要对特定类别应用模糊。"""
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)
        color = colors[class_id]
        B, G, R = map(int, color)

        # 如果第一次看到 track_id，则分配新的类别特定ID
        if track_id not in track_class_mapping:
            class_counters[class_id] += 1
            track_class_mapping[track_id] = class_counters[class_id]

        class_specific_id = track_class_mapping[track_id]
        text = f"{class_specific_id} - {class_names[class_id]}"

        # 绘制矩形和文本
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(
            frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1
        )
        cv2.putText(
            frame,
            text,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # 对指定类别应用高斯模糊
        if FLAGS.class_blur_id is not None and class_id == FLAGS.class_blur_id:
            if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

    return frame


def main(argv):
    logger.info(f"脚本名称: {argv[0]}")
    try:
        cap = initialize_video_capture(FLAGS.inputs)
        model = initialize_model(FLAGS.model_path)
        class_names = load_class_names(FLAGS.classes_path)

        # 获取视频属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        writer = cv2.VideoWriter(
            FLAGS.outputs, fourcc, fps, (frame_width, frame_height)
        )

        # 初始化跟踪器
        tracker = DeepSort(max_age=20, n_init=3)

        # 为每个类别设置随机颜色
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3))

        class_counters = defaultdict(int)
        track_class_mapping = {}
        frame_count = 0

        while True:
            start = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                break

            # 处理并跟踪帧中的对象
            tracks = process_frame(frame, model, tracker, class_names, colors)
            frame = draw_tracks(
                frame, tracks, class_names, colors, class_counters, track_class_mapping
            )

            end = datetime.datetime.now()
            logger.info(
                f"处理帧 {frame_count} 的时间：{(end - start).total_seconds():.2f} 秒"
            )
            frame_count += 1

            # 在帧上显示FPS
            fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(
                frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8
            )

            # 将处理后的帧写入输出视频
            writer.write(frame)
            cv2.imshow(f"{type(model).__name__} object tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 输出每个类别的计数
        logger.info("类别计数：")
        for class_id, count in class_counters.items():
            logger.info(f"{class_names[class_id]}: {count}")

    except Exception:
        logger.exception("处理过程中发生错误")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit as e:
        logger.info(f"捕获到 SystemExit 异常：{e}")
    except Exception:
        logger.exception("主程序运行过程中发生未捕获的异常")
