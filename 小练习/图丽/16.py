import json
import uuid

import yaml

# 导入数据
json_file_path = "default_config.json"
with open(json_file_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)

aim_data = {"spaces": []}

for region in json_data["veh_attr_param"]["attributing_regions"]:
    # 为每个区域生成一个唯一的 space_id
    space_id = str(uuid.uuid4())

    # 提取和转换坐标
    region_coords = []
    for point in region["ps"]:
        region_coords.extend([point["x"], point["y"]])

    # 添加到新数据结构
    aim_data["spaces"].append({"space_id": space_id, "region": region_coords})

# 将数据转换为 YAML 格式
yaml_data = yaml.safe_dump(
    aim_data, allow_unicode=True, default_flow_style=False, sort_keys=False
)

# 保存 YAML 数据到文件
yaml_file_path = "default_config.yaml"
with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
    yaml_file.write(yaml_data)
