import matplotlib.pyplot as plot


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


# 定义多边形和点
polygon_coords = [369, 946, 1, 812, 2, 1064, 197, 1066]
polygon_points = [
    (polygon_coords[i], polygon_coords[i + 1]) for i in range(0, len(polygon_coords), 2)
]
point = (100, 900)


# 判断
import time

t_s = time.time()
for _ in range(10):
    if is_point_inside_polygon(point, polygon_points):
        print("The point is inside the polygon.")
    else:
        print("The point is outside the polygon.")
t_e = time.time()
print((t_e - t_s) / 10)


# 提取x和y坐标
x_coords, y_coords = zip(*polygon_points)

# 创建一个新的图形
plot.figure()

# 绘制多边形
plot.plot(x_coords + (x_coords[0],), y_coords + (y_coords[0],), marker="o")

# 绘制点
plot.plot(point[0], point[1], marker="o", color="red")

# 添加标签
plot.text(
    point[0],
    point[1],
    f"({point[0]}, {point[1]})",
    fontsize=12,
    verticalalignment="bottom",
)

# 设置标题和坐标轴标签
plot.title("Polygon and Point")
plot.xlabel("X Coordinate")
plot.ylabel("Y Coordinate")

# 显示图形
plot.show()
