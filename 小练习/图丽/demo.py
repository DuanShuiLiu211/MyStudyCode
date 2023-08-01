import subprocess
import time
import re
import os


def parse_output(output):
    # 提取视频标题
    title_match = re.search(r'title:\s+(.*)', output)
    title = title_match.group(1) if title_match else ''
    title = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", "", title)

    # 提取流的信息
    format_matches = re.findall(
        r'- format:\s+\x1b\[7m(.+?)\x1b\[0m\n\s+container:\s+(.+?)\n\s+quality:\s+(.+?)\n\s+size:\s+(.+?)\n',
        output, re.DOTALL)
    formats = []
    for match in format_matches:
        f = {
            'format': match[0],
            'container': match[1],
            'quality': match[2].replace(" ", ""),
            'size': match[3]
        }
        formats.append(f)

    return title, formats


def scp_file(source_path, destination_path, password):
    command = f"scp -r '{source_path}' '{destination_path}'"

    p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 向标准输入流(stdin)发送密码
    p.stdin.write(password.encode('utf-8'))
    p.stdin.close()

    # 获取命令执行的输出结果
    output, error = p.communicate()

    # 打印输出结果
    print(output.decode('utf-8'))


def download_video(url, cookie, output_dir="./"):
    command = f"you-get -i '{url}' -c '{cookie}'"
    output = subprocess.check_output(command, shell=True).decode('utf-8')

    # 解析输出字符串
    title, formats = parse_output(output)

    if formats:
        # 定义优先级顺序的格式列表
        priority_formats = [
            'dash-hdflv2_4k', 'dash-hdflv2', 'dash-flv', 'dash-flv720',
            'dash-flv480', 'dash-flv360', 'flv360'
        ]

        # 按照优先级选择可用的格式
        selected_format = None
        for format in priority_formats:
            for f in formats:
                if format == f['format']:
                    selected_format = f
                    break
            if selected_format:
                break

        if selected_format:
            # 提取选定格式的URL进行下载
            format = selected_format['format']
            quality = selected_format['quality']
            timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
            output_dir = os.path.join(output_dir, quality, title)
            os.makedirs(output_dir, exist_ok=True)
            command = f"you-get '{url}' -c '{cookie}' --format='{format}' -o '{output_dir}' -O '{timestamp}.mp4'"
            completed_process = subprocess.run(command, shell=True)
            # 检查运行结束的状态码
            if completed_process.returncode == 0:
                print("下载完成")
                scp_file(f"{output_dir}", 'tlkj@192.168.3.36:/home/tlkj/datas/wangh/rawdatas/internet_video/', 'Tlkj@passw0rd')
            else:
                print("下载出错")
        else:
            print('No suitable video format found.')
    else:
        print('No video formats found.')


if __name__ == '__main__':
    video_info_list = []
    with open("./assets/视频信息列表.txt", "r") as text_file:
        for line in text_file:
            video_info_list.append(line.strip())

    select_video_info_list = []
    keywords_place = ['青', '新', '藏']
    keywords_other = ['车', '公路', '国道']
    for idx, video_info in enumerate(video_info_list):
        if any([i in video_info for i in keywords_place]) and any(
            [i in video_info for i in keywords_other]):
            select_video_info_list.append(video_info)

    for video_info in select_video_info_list:
        pattern = r'(https://www\.bilibili\.com/video/av\d+)'
        match = re.search(pattern, video_info)
        if match:
            url = match.group(1)
            cookie = '/Users/WangHao/Desktop/TODO/cookies.sqlite'
            output_dir = "./"
            download_video(url, cookie, output_dir)
        else:
            print("Video link not found.")
