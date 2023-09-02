import threading
import asyncio
import datetime
import time


"""
0. async 定义的函数是协程函数调用协程函数返回协程对象
1. await 可以处理可等待对象，可等待对象有协程对象（corotine）、任务对象（task，用于顶层操作，必须作用于协程对象）和未来对象（future，底层操作 task 的父类）
2. 协程的核心是事件循环，目的是合理的调度可等待对象的执行逻辑从而优化io等待时间
2. 事件循环的最小调度单元是任务，通过主动的将协程对象封装成的协程任务并注册到事件循环并通过内部的调度算法调度执行
3. 运行协程函数需要使用特定的接口调用如 asyncio.run(main())，它会启动一个事件循环并将 main() 作为任务注册到事件循环中调度执行
"""


# example 1
async def display_date():
    loop = asyncio.get_running_loop()
    end_time = loop.time() + 5.0
    while True:
        print(datetime.datetime.now())
        print(loop.time() + 1.0)
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(1)  # 可等待对象将
        print(datetime.datetime.now())


# 1. asyncio.run(display_date()) 创建了一个事件循环并将中 display_date() 作为 task 注册
# 2. display_date() 有一个可等待对象（awaitable） asyncio.sleep(1) 执行就会阻塞当前 task 1秒，阻塞过程中由于没有其他的 task 所以 cpu 也在等待
asyncio.run(display_date())


async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)


async def main():
    print(f"started at {time.strftime('%X')}")

    await say_after(1, 'hello')
    await say_after(2, 'world')

    print(f"finished at {time.strftime('%X')}")


# 1. 使用 async 创建协程函数 main() 和 say_after(delay, what)
# 2. 使用 asyncio.run(main()) 创建了一个事件循环（eventloop）并将中 main() 作为第一个 task 调度执行
# 3. 使用了 await 创建了可等待对象 say_after(1, 'hello') 和 say_after(2, 'hello')
# 4. 可以发现当前事件循环中只有 main() 无法异步调度，并且 main() 中有两个 awaitable 存在异步调度的改进空间
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


# 执行这段协程程序的逻辑如下
# 1. 使用 async 创建协程函数 say_after(delay, what) 和 main()。
# 2. 使用 asyncio.run(main()) 创建一个新的事件循环，并将程序的控制权交给这个事件循环。
# 3. main() 将作为事件循环的第一个任务，获得程序的控制权开始执行。虽然它是在事件循环上下文中执行的，但直到遇到一个 await 语句，控制权都不会返回给事件循环。
# 4. 在 main() 内部，使用 asyncio.create_task() 创建了两个任务 task1 和 task2。这两个任务被添加到事件循环的任务队列中，并标记为准备执行。事件循环被通知了这些任务，但直到 main() 释放控制权，事件循环才真正开始执行它们。
# 5. 在创建完这两个任务之后，main() 继续执行 print() 语句。
# 6. 当 main() 执行到 await task1 时，它将控制权交还给事件循环，等待 task1 完成。事件循环此时开始执行队列中的任务。task1 和 task2 几乎同时开始，并都进入它们各自的 await asyncio.sleep() 语句。
# 7. 在 task1 的休眠时间结束后，它继续执行并打印 "hello"。此时 main() 中的 await task1 也完成了，所以 main() 继续执行，直到下一个 await，即 await task2。
# 8. await task2 需要等待 task2 完成。但因为 task2 已经开始了，并且已经等待了大约1秒，所以此时只需再等待约1秒。
# 9. task2 完成其休眠后，继续执行并打印 "world"。此时 main() 中的 await task2 也完成了。
# 10. main() 打印 "finished" 信息。
# 11. main() 完成执行，程序控制权返回给事件循环。但事件循环中没有更多的任务要执行，所以它结束，并随之结束了整个程序。
asyncio.run(main())  # 花费时间为2秒


# 协程函数的并发是在一个线程内通过事件循环实现的，因此一个线程在同一时间也只能有一个活跃的事件循环
# 1. asyncio.wait() set 形式返回任务执行的详情以及future值（即协程最后return的值）
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
    result = await asyncio.wait([hello1(), hello2(), hello3()])
    print(result)
    time_end = time.time()
    print(f"Take time: {time_end - time_start}")


asyncio.run(main())


# 2. asyncio.gather() 列表形式返回 future值
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
    result = await asyncio.gather(hello1(), hello2(), hello3())  # 使用gather来并发执行协程
    print(result)
    time_end = time.time()
    print(f"Take time: {time_end - time_start}")


asyncio.run(main())
