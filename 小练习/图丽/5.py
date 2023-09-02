import asyncio

async def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
        await asyncio.sleep(0.1) 

async def print_dots():
    for _ in range(20):  # 打印20个点
        print(".", end="", flush=True)
        await asyncio.sleep(0.05)

async def main():
    task1 = asyncio.create_task(print_dots())
    
    # 直接在这里迭代fib函数
    async for value in fib(10):
        print(value)
    
    # 等待print_dots完成
    await task1

asyncio.run(main())
