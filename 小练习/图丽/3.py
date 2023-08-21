# 协程的核心是事件循环，将协程函数注册到事件循环中让事件循环去统一的调度协程函数
# 1. 运行一个协程函数
import threading
import asyncio
import time


async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)


async def main():
    print(f"started at {time.strftime('%X')}")

    await say_after(1, 'hello')
    await say_after(2, 'world')

    print(f"finished at {time.strftime('%X')}")


asyncio.run(main())  # asyncio.run()将使用事件循环去统一调度协程函数，因此花费时间为3秒和同步执行


async def main():
    task1 = asyncio.create_task(say_after(1, 'hello'))
    task2 = asyncio.create_task(say_after(2, 'world'))

    print(f"started at {time.strftime('%X')}")

    await task1
    await task2

    print(f"finished at {time.strftime('%X')}")


asyncio.run(main())  # 创建task会将协程函数加入事件循环，因此花费为2秒实现了异步执行


async def hello1():
    print(f"Hello world 01 begin,my thread is: {threading.currentThread()}")
    await asyncio.sleep(3)
    print("Hello again 01 end")


async def hello2():
    print(f"Hello world 02 begin,my thread is: {threading.currentThread()}")
    await asyncio.sleep(2)
    print("Hello again 02 end")


async def hello3():
    print(f"Hello world 03 begin,my thread is: {threading.currentThread()}")
    await asyncio.sleep(1)
    print("Hello again 03 end")


async def main():
    time_start = time.time()
    tasks = [hello1(), hello2(), hello3()]
    await asyncio.wait(tasks)
    time_end = time.time()
    print(f"Take time: {time_end - time_start}")


asyncio.run(main())
