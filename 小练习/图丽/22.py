import multiprocessing
import signal

event = multiprocessing.Event()

def signal_handler(signal_number, code_frame):
    print("Signal handler called with signal", signal_number)
    print("Code frame:", code_frame)
    event.set()


def child_process(event):
    print("Press Ctrl+C to trigger the signal handler...")
    while not event.is_set():
        print("Waiting for signal...")
        event.wait(timeout=1)


def main():
    # 设置信号处理程序
    signal.signal(signal.SIGINT, signal_handler)

    # 创建多个子进程
    processes = []
    for _ in range(5):
        process = multiprocessing.Process(target=child_process, args=(event,))
        processes.append(process)

    # 启动子进程
    for process in processes:
        process.start()

    # 等待所有子进程结束
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
    print("Exiting…")
