def group_adjacent_points(points):
    # 根据点的坐标进行排序
    sorted_points = sorted(points)

    # 初始化用于存储分组的列表
    groups = []

    # 初始化当前分组
    current_group = [sorted_points[0]]

    # 遍历排序后的点列表，将相邻的点组合在一起
    for point in sorted_points[1:]:
        prev_point = current_group[-1]
        if is_adjacent(prev_point, point):
            current_group.append(point)
        else:
            groups.append(current_group)
            current_group = [point]

    # 添加最后一个分组
    groups.append(current_group)

    return groups


def is_adjacent(point1, point2):
    # 检查两个点是否相邻
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1


# 示例数据点
points = [(3, 4), (3, 3), (1, 1), (4, 2), (4, 4), (5, 5), (5, 6)]

# 将相邻的点组合在一起
groups = group_adjacent_points(points)

# 打印分组结果
for group in groups:
    print(group)
