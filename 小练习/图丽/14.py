import requests
from bs4 import BeautifulSoup
import os

# 目标网址
url = 'https://sale.1688.com/factory/card.html?__existtitle__=1&__removesafearea__=1&memberId=b2b-22118751966000455c&aHdkaW5n_isCentral=true&aHdkaW5n_isGrayed=false&aHdkaW5n_isUseGray=true&topOfferIds=745601621093&clickArea=body&query=%E5%8D%97%E9%80%9A%E5%9F%8E%E9%82%A6&spm=a26352.b40562653.offer.0&src_cna=uFLrHWg0fTwBASQOBHkQuuQz'

# 用户代理
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# 发送GET请求
response = requests.get(url, headers=headers)

# 解析HTML内容
soup = BeautifulSoup(response.text, 'html.parser')

# 查找所有图片标签
images = soup.find_all('img')

# 图片保存路径
save_path = './images'

# 确保保存路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 遍历并下载每张图片
for img in images:
    img_url = img.get('src')
    if img_url:
        # 处理相对URL
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif img_url.startswith('/'):
            img_url = url.rsplit('/', 1)[0] + img_url

        try:
            # 发送请求获取图片数据
            img_data = requests.get(img_url, headers=headers).content
            # 提取图片文件名
            filename = os.path.basename(img_url.split('?')[0])  # 移除URL参数
            # 保存图片
            with open(os.path.join(save_path, filename), 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f"无法下载图片 {img_url}: {e}")
