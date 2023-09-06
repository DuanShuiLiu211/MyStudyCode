import signal
import time
import os


print("Current process ID:", os.getpid())
print(signal.SIGSTOP.value)


def signal_handler(sig, frame):
    print(f"Received signal: {sig}")
    print(f"Current function name in stack frame: {frame.f_code.co_name}")


signal.signal(signal.SIGINT, signal_handler)  # 对应 Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 默认的 kill 信号

print("Starting... Press Ctrl+C or kill -2 PID to trigger the signal.")
time.sleep(100)
