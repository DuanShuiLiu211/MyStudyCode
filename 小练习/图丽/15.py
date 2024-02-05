message = """
# a1.py
class abc:
    def __init__(self):
        self.a = [1]


a = abc()
b = 2
print(a.a, b)
print(id(a.a), id(b))


# a2.py
from a1 import a, b

print(a.a, b)
print(id(a.a), id(b))

a.a.append(2)
b = 1

print(a.a, b)
print(id(a.a), id(b))

from a1 import a, b

print(a.a, b)
print(id(a.a), id(b))


模块中变量的初始状态：在 a1.py 中，变量 b 被初始化为 2。

第一次导入：当您在 a2.py 中第一次执行 from a1 import a, b 时，您实际上是将 a1.py 中的 a 和 b 的当前值导入到 a2.py 的命名空间。此时，a2.py 中的 b 变成了值 2。

在 a2.py 中修改 b：当您在 a2.py 中执行 b = 1，您实际上是改变了 a2.py 中 b 的值。这个操作并不影响 a1.py 中 b 的值。

第二次导入：当您再次执行 from a1 import a, b，Python 会再次从 a1.py 中将 a 和 b 的值导入到 a2.py。对于 a 来说，由于它是一个可变对象（列表），之前对它的修改（a.a.append(2)）影响了 a1.py 中 a 的状态，所以这个状态被保留下来。但对于 b 来说，由于它是一个不可变对象，a2.py 中对 b 的修改不会反映到 a1.py 上，因此再次导入时，a1.py 中 b 的原始值（2）会重新被导入到 a2.py。

总结来说，这种现象发生是因为再次执行 from a1 import a, b 时，不可变类型的变量 b 被重新设置为 a1.py 中定义的值。这是 Python 中从模块导入变量时的标准行为。对于不可变对象，如整数、字符串和元组，再次从模块导入它们会导致在当前命名空间中使用模块内部定义的原始值。
"""

print(message)
