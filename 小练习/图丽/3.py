import threading
import asyncio
import datetime
import time


# 协程的核心是事件循环，将协程函数注册到事件循环中让事件循环去统一的调度协程函数
# 运行协程函数需要使用特定的接口调用如asyncio.run()（类似生成器直接调用并不会执行）
# 默认情况下的协程函数与一般的函数执行没有区别，代码仅仅同步的阻塞式运行
async def display_date():
    loop = asyncio.get_running_loop()
    end_time = loop.time() + 5.0
    while True:
        print(datetime.datetime.now())
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(1)
        print(datetime.datetime.now())


asyncio.run(display_date())


# 当使用asyncio.create_task()对协程函数进行封装后，代码将异步的并发运行
async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)


async def main():
    print(f"started at {time.strftime('%X')}")

    await say_after(1, 'hello')
    await say_after(2, 'world')

    print(f"finished at {time.strftime('%X')}")


asyncio.run(main())  # 花费时间为3秒


async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)


async def main():
    task1 = asyncio.create_task(say_after(1, 'hello'))
    task2 = asyncio.create_task(say_after(2, 'world'))
    print(f"started at {time.strftime('%X')}")

    await task1
    await task2

    print(f"finished at {time.strftime('%X')}")


asyncio.run(main())  # 花费时间为2秒


# 协程函数的并发是在一个线程内通过事件循环实现的，因此一个线程在同一时间也只能有一个活跃的事件循环
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