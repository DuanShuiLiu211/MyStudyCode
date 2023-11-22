def function(a, b):
    if a > 1:
        print("a ok")
        return a
    if b > 1:
        print("b ok")
        return b


def nochange(a):
    print(a, id(a))  # a指向对象1
    a = 0
    print(a, id(a))  # a一个新对象2


def change(list):
    print(id(list))
    list.append([4, 5, 6])
    print("函数内: ", list, id(list))
    return


def need(str, a=1):
    print(str)
    print(a)
    return


def more(g, **vartuple):
    print(g)
    print(vartuple)


def mix(a, /, b, *, c):
    return print(a + b + c)


sum = lambda arg1, arg2: arg1 + arg2

import time


def z(f):
    def y(x):
        start_time = time.time()  # 开始时间
        f(x)  # 运行输入的函数
        stop_time = time.time()  # 结束时间
        delta_time = stop_time - start_time  # 运行时间
        return print("{:.9f}".format(delta_time))

    return y


@z
def t(a):
    return a**9 * a**9


if __name__ == "__main__":
    c = function(0, 3)
    print(c)

    a = 1
    print(a, id(a))  # a指向对象1
    nochange(a)
    print(a, id(a))  # a指向对象1

    list = [1, 2, 3]
    d = change(list)
    print(d)
    print("函数外: ", list, id(list))

    # need()
    need(1)

    more(1, b=2, c=3, d=4, e=5)

    mix(1, 2, c=3)

    print(sum(10, 20))

    t(25000)
