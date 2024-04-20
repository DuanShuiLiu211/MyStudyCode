# 海象运算符是先计算右边的表达式，然后再将结果赋值给左边的变量。
if positive := (2 > 1):
    if positive := (3 > 4):
        pass

print(positive)
