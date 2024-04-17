import multiprocessing
import signal


def child_process(flag):
    # 注册信号处理函数
    def signal_handler(signal_number, code_frame):
        # 在这里定义当信号被触发时要执行的操作
        print("Signal handler called with signal", signal_number)
        print("Code frame:", code_frame)
        flag.value = False

    signal.signal(signal.SIGINT, signal_handler)  # 注册 Ctrl+C 信号处理程序

    # 无限循环来保持程序运行，直到 flag 的值为 False
    print("Press Ctrl+C to trigger the signal handler...")
    while flag.value:
        print("Waiting for signal...")


def main():
    # 创建共享标志变量
    flag = multiprocessing.Value("b", True)

    # 创建多个子进程
    processes = []
    for _ in range(5):
        process = multiprocessing.Process(target=child_process, args=(flag,))
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
