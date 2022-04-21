import pyautogui
import threading
import keyboard
import time


flag = True


def watch_key(key):
    name = key.name
    print(name)
    global flag
    if name == 'esc':
        flag = False
        # print('此时flag为False，将退出子线程main')
    elif name == 'delete':
        # print('此时执行delete，删除一个目标')
        pass
    else:
        print('想退出子线程main，请按esc')


class MyThread(threading.Thread):
    def __init__(self, thread_id, name):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name

    def run(self):
        print("开始线程: " + self.name)
        main()
        print("退出线程: " + self.name)


def CurrentMousePlace():
    pyautogui.alert(text='请先选中对话框非确认按钮区域\n再将鼠标移至目标上方\n最后回车确认', title='命令')
    CurrentMouseW, CurrentMouseH = pyautogui.position()
    return CurrentMouseW, CurrentMouseH


def MouseOperationCl(wight, height, time=1.0):
    if pyautogui.onScreen(wight, height):
        pyautogui.moveTo([wight, height], duration=time)
        pyautogui.click(button='left')
    else:
        print('当前屏幕上没有鼠标')


def MouseOperationDcl(wight, height, time=1.0):
    if pyautogui.onScreen(wight, height):
        pyautogui.moveTo([wight, height], duration=time)
        pyautogui.doubleClick(button='left')
    else:
        print('当前屏幕上没有鼠标')


def MouseOperationClPd(wight, height, time=1.0):
    if pyautogui.onScreen(wight, height):
        pyautogui.moveTo([wight, height], duration=time)
        pyautogui.click(button='left')
        pyautogui.press('Delete')
    else:
        print('当前屏幕上没有鼠标')


def step_one():
    w1, h1 = CurrentMousePlace()
    print(w1, h1)
    w2, h2 = CurrentMousePlace()
    print(w2, h2)
    w3, h3 = CurrentMousePlace()
    print(w3, h3)
    w4, h4 = CurrentMousePlace()
    print(w4, h4)
    return w1, h1, w2, h2, w3, h3, w4, h4


def step_two(w1, h1, w2, h2, w3, h3, w4, h4):
    MouseOperationDcl(w1, h1, time=0.5)
    time.sleep(1.5)

    MouseOperationDcl(w2, h2, time=0.5)
    time.sleep(1.5)

    MouseOperationCl(w3, h3, time=0.5)
    time.sleep(1.5)

    MouseOperationCl(w4, h4, time=0.5)
    time.sleep(1.5)


def step_three(w1, h1, w2, h2, w3, h3, w4, h4):
    MouseOperationDcl(w1, h1, time=0.25)
    time.sleep(0.75)

    MouseOperationClPd(w2, h2, time=0.5)
    MouseOperationDcl(w2, h2, time=0.25)
    MouseOperationCl(w3, h3, time=0.25)
    time.sleep(0.5)

    MouseOperationCl(w4, h4, time=0.25)


def main():
    print(pyautogui.size())
    keyboard.on_press(watch_key)  # 若使用hook则按下与释放都会触发
    if flag:        w1, h1, w2, h2, w3, h3, w4, h4 = step_one()
    if flag:
        pyautogui.alert(text='准备开始', title='提示')
    if flag:
        step_two(w1, h1, w2, h2, w3, h3, w4, h4)
    if flag:
        for i in range(1, 1500):
            if flag:
                step_three(w1, h1, w2, h2, w3, h3, w4, h4)


if __name__ == '__main__':
    start_time = time.time()
    thread_main = MyThread(1, 'main')
    thread_main.start()  # 开始子线程main
    thread_main.join()  # 当线程中止时，退出子线程main
    end_time = time.time()
    sum_time = end_time - start_time
    print('程序运行时间：{:.2f}秒'.format(sum_time))

