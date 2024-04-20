import argparse
import subprocess
import logging
import time

# 服务持续时间
MAX_DURATION = 10 * 60

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="/home/wanghao/data/lane_gate/in_out/in/in_20240124_165600_26.mkv",
        help="输入文件",
    )
    parser.add_argument(
        "--output_url",
        default="rtsp://localhost:8554/mystream",
        help="输出链接",
    )

    return parser.parse_args()


def start_rtsp_stream(input_file, output_url):
    """
    启动 RTSP 流

    :param input_file: 输入视频文件的路径
    :param output_url: 输出 RTSP 流的 URL
    """
    command = [
        "ffmpeg",
        "-stream_loop",
        "-1",  # 无限循环输入文件
        "-re",  # 以原始帧速率处理
        "-i",
        input_file,
        "-c",
        "copy",
        "-f",
        "rtsp",  # 使用 RTSP 格式输出
        "-rtsp_transport",
        "tcp",  # 使用 TCP 协议传输
        output_url,
    ]

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(f"启动 RTSP 流：{input_file} -> {output_url}")
        return process
    except Exception as e:
        logging.error(f"启动 RTSP 流失败：{e}")
        exit(1)


def stop_rtsp_stream(process) -> None:
    """
    停止 RTSP 流

    :param process: RTSP 流的进程对象
    """
    try:
        process.terminate()
        process.wait(timeout=30)
        logging.info("RTSP 流已停止")
    except subprocess.TimeoutExpired:
        logging.warning("停止 RTSP 流超时强制终止")
        process.kill()
    except Exception as e:
        logging.error(f"停止 RTSP 流时发生错误：{e}")


def main():
    args = get_args()
    input_file = args.input_file
    output_url = args.output_url

    # 启动 RTSP 流
    process = start_rtsp_stream(input_file, output_url)

    # 等待一段时间
    time.sleep(MAX_DURATION)

    # 停止 RTSP 流
    stop_rtsp_stream(process)


if __name__ == "__main__":
    main()
