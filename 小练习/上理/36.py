# 初始化字典
a = {"b": 5, "c": 2, "a": 4, "d": 1}

# 对字典按键（key）进行排序（默认由小到大）
sort_key = sorted(a.items(), key=lambda x: x[0])
# 输出结果
print(sort_key)  # [('a', 4), ('b', 5), ('c', 2), ('d', 1)]

# 对字典按值（value）进行排序（默认由小到大）
sort_value = sorted(a.items(), key=lambda x: x[1])
# 输出结果
print(sort_value)  # [('d', 1), ('c', 2), ('a', 4), ('b', 5)]
