import multiprocessing
import os
import signal
import time


def signal_handler(event):
    def handler(signal_number, code_frame):
        print("进程号", os.getpid(), "事件号", id(event), "事件状态", event.is_set())
        print("信号类型", signal_number)
        print("代码帧", code_frame)
        event.set()
        print("进程号", os.getpid(), "事件号", id(event), "事件状态", event.is_set())

    return handler


def child_process(event):
    signal.signal(signal.SIGINT, signal_handler(event))
    signal.signal(signal.SIGTERM, signal_handler(event))
    event.wait(5)
    while not event.is_set():
        print("子进程号", os.getpid(), "事件号", id(event), "事件状态", event.is_set())
        time.sleep(1)


def main():
    event = multiprocessing.Event()
    signal.signal(signal.SIGINT, signal_handler(event))
    signal.signal(signal.SIGTERM, signal_handler(event))
    print("主进程号", os.getpid(), "事件号", id(event))
    processes = []
    for _ in range(5):
        process = multiprocessing.Process(target=child_process, args=(event,))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    print("程序开始")
    main()
    print("程序结束")
