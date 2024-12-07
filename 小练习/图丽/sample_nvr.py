"""
app_config.json = {
    "sources": [{"name": "x", "url": "xxxxx"},
                {"name": "y", "url": "yyyyy"}],
    "max_duration": 600,
    "split_interval": 60,
}
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime

BASE_PATH: str = os.path.abspath(os.path.dirname(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=os.path.join(BASE_PATH, "assets/config/app_config.json"),
        help="程序的主要配置文件",
    )

    return parser.parse_args()


def validate_parameters(config_dict):
    max_duration = config_dict.get("max-duration", 30 * 60)
    if not isinstance(max_duration, int):
        print(f"max-duration is not configured as int, assume to 30 minutes")
        max_duration = 30 * 60

    split_interval = config_dict.get("split-interval", 0)
    if not isinstance(split_interval, int):
        print("split-interval is not configured as int, assume to 0 minutes")
        split_interval = 0

    return max_duration, split_interval


def generate_ffmpeg_commands(config_dict, max_duration, split_interval):
    ffmpeg_commands = []

    for source in config_dict["sources"]:
        url = source["url"]
        ffmpeg_cmd = ["ffmpeg", "-i", url, "-rtsp_transport", "tcp", "-c", "copy"]

        if max_duration > 0:
            ffmpeg_cmd.extend(["-t", str(max_duration)])

        if split_interval > 0:
            ffmpeg_cmd.extend(
                [
                    "-f",
                    "segment",
                    "-segment_time",
                    str(split_interval),
                    "-reset_timestamps",
                    "1",
                ]
            )
        name = source["name"]
        postfix: str = datetime.now().strftime("%Y年%m月%d日%H点%M分%S秒")
        ffmpeg_cmd.extend([f"{name}_%04d_{postfix}.mp4"])
        print(ffmpeg_cmd)
        ffmpeg_commands.append(ffmpeg_cmd)

    return ffmpeg_commands


def check_stderr_for_disconnect(stderr):
    # 在 stderr 中搜索特定的字符串来检测连接状态
    # 这里假设 "connection closed" 是一个表示连接断开的消息
    return "connection closed" in stderr.lower()


def main() -> None:
    args = get_args()
    config_path: str = args.config_path
    assert os.path.exists(config_path), f"config file not found: {config_path}"

    with open(file=config_path, mode="r") as f:
        config_dict = json.load(f)
    print(config_dict)

    max_duration, split_interval = validate_parameters(config_dict)
    ffmpeg_cmds = generate_ffmpeg_commands(config_dict, max_duration, split_interval)

    while True:
        ffmpeg_sps = []
        for ffmpeg_cmd in ffmpeg_cmds:
            ffmpeg_sp = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            ffmpeg_sps.append(ffmpeg_sp)

        for ffmpeg_sp in ffmpeg_sps:
            ffmpeg_sp.wait()
            returncode = ffmpeg_sp.returncode
            stdout, stderr = ffmpeg_sp.communicate()
            print(f"returncode: {returncode}")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")

        if returncode == 0:
            break
        else:
            print("Connection lost. Reconnecting...")
            time.sleep(10)


if __name__ == "__main__":
    main()
