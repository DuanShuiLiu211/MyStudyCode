def calculate_area(bbox):
    x1, y1, x2, y2 = bbox
    return abs(x2 - x1) * abs(y2 - y1)


# 示例字典
detected_results = {
    "obj1_0": ([10, 20, 30, 50], 0.9),
    "obj2_1": ([5, 5, 25, 45], 0.8),
    "obj3_2": ([15, 25, 35, 55], 0.95),
}

# 按面积排序
sorted_results = sorted(
    detected_results.items(), key=lambda x: calculate_area(x[1][0]), reverse=True
)

for key, value in sorted_results:
    print(key, calculate_area(value[0]))
