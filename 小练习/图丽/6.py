import asyncio
import time


async def say_after(delay, what):
    print(f"flag at {time.strftime('%X')}")
    await asyncio.sleep(delay)
    print(what)


async def main():
    task1 = asyncio.create_task(say_after(1, "hello"))
    print(task1, type(task1))
    task2 = asyncio.create_task(say_after(2, "world"))
    print(task2, type(task2))
    print(f"started at {time.strftime('%X')}")

    await task1
    print(task1, type(task1))
    print(f"task1 at {time.strftime('%X')}")
    await task2
    print(task2, type(task2))
    print(f"task2 at {time.strftime('%X')}")

    print(f"finished at {time.strftime('%X')}")


asyncio.run(main())  # 花费时间为2秒
