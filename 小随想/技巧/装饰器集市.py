import functools
import time


# 一个使用条件生成装饰器的例子
def take_time(threshold):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            if end_time - start_time > threshold:
                print(f"{func.__name__}函数运行时间超过了{threshold}秒")

        return wrapper

    return decorator


@take_time(1)
def sleep(secs):
    time.sleep(secs)


sleep(2)
