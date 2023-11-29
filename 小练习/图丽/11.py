import random


def is_point_inside_polygon(point, polygon_points):
    x, y = point
    odd_nodes = False
    j = len(polygon_points) - 1

    for i in range(len(polygon_points)):
        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]
        if yi < y and yj >= y or yj < y and yi >= y:
            if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                odd_nodes = not odd_nodes
        j = i

    return odd_nodes


def bounding_box(polygon):
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def overlap_area(polygon1, polygon2, samples=10000):
    # 获取两个多边形的边界框
    bb1 = bounding_box(polygon1)
    bb2 = bounding_box(polygon2)

    # 计算两个边界框的交集
    bb_overlap = (
        max(bb1[0], bb2[0]),
        max(bb1[1], bb2[1]),
        min(bb1[2], bb2[2]),
        min(bb1[3], bb2[3]),
    )

    # 如果边界框没有交集，那么重叠面积为0
    if bb_overlap[2] <= bb_overlap[0] or bb_overlap[3] <= bb_overlap[1]:
        return 0

    count_inside_both = 0
    for _ in range(samples):
        x = random.uniform(bb_overlap[0], bb_overlap[2])
        y = random.uniform(bb_overlap[1], bb_overlap[3])

        if is_point_inside_polygon((x, y), polygon1) and is_point_inside_polygon(
            (x, y), polygon2
        ):
            count_inside_both += 1

    return (
        count_inside_both
        / samples
        * (bb_overlap[2] - bb_overlap[0])
        * (bb_overlap[3] - bb_overlap[1])
    )


polygon1 = [(369, 946), (1, 812), (2, 1064), (197, 1066)]
polygon2 = [(400, 900), (20, 850), (10, 1000), (150, 1100)]

import time

t_s = time.time()
for _ in range(10):
    area = overlap_area(polygon1, polygon2)
    print(f"Estimated overlap area: {area}")
t_e = time.time()
print((t_e - t_s) / 10)


from shapely.geometry import Polygon


def overlap_area(coords1, coords2):
    # 将坐标列表转换为点元组的列表
    polygon1 = Polygon(
        [(coords1[i], coords1[i + 1]) for i in range(0, len(coords1), 2)]
    )
    polygon2 = Polygon(
        [(coords2[i], coords2[i + 1]) for i in range(0, len(coords2), 2)]
    )

    # 计算交集面积
    return polygon1.intersection(polygon2).area


polygon_coords1 = [369, 946, 1, 812, 2, 1064, 197, 1066]
polygon_coords2 = [400, 900, 20, 850, 10, 1000, 150, 1100]

import time

t_s = time.time()
for _ in range(10):
    area = overlap_area(polygon_coords1, polygon_coords2)
    print(f"Overlap area: {area}")
t_e = time.time()
print((t_e - t_s) / 10)
