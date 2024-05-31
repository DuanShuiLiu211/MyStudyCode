import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

xml_path = "/Users/WangHao/Desktop/TODO/测试/黑L4635挂.xml"
tree = ET.parse(xml_path)
root = tree.getroot()

image_path = root.find("path").text
image = Image.open(image_path)

draw = ImageDraw.Draw(image)
for obj in root.findall("object"):
    xmin = int(obj.find("bndbox/xmin").text)
    ymin = int(obj.find("bndbox/ymin").text)
    xmax = int(obj.find("bndbox/xmax").text)
    ymax = int(obj.find("bndbox/ymax").text)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red")

output_image_path = "output_image_with_boxes.jpg"
image.save(output_image_path)

print("带有边界框的图片已保存为:", output_image_path)
