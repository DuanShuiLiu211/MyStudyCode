import pyautogui
import time


def CurrentMousePlace():
    pyautogui.alert(text='请先选中对话框非确认按钮区域\n再将鼠标移至目标上方\n最后回车确认', title='命令')
    CurrentMouseW, CurrentMouseH = pyautogui.position()
    return CurrentMouseW, CurrentMouseH


def MouseOperationCl(wight, height, time=1.0):
    flag = pyautogui.onScreen(wight, height)

    if flag:
        pyautogui.moveTo([wight, height], duration=time)
        pyautogui.click(button='left')
    else:
        print('当前屏幕上没有鼠标')


def MouseOperationDcl(wight, height, time=1.0):
    flag = pyautogui.onScreen(wight, height)

    if flag:
        pyautogui.moveTo([wight, height], duration=time)
        pyautogui.doubleClick(button='left')
    else:
        print('当前屏幕上没有鼠标')


def MouseOperationClPd(wight, height, time=1.0):
    flag = pyautogui.onScreen(wight, height)

    if flag:
        pyautogui.moveTo([wight, height], duration=time)
        pyautogui.click(button='left')
        pyautogui.press('Delete')
    else:
        print('当前屏幕上没有鼠标')


def main():
    w1, h1 = CurrentMousePlace()
    print(w1, h1)
    w2, h2 = CurrentMousePlace()
    print(w2, h2)
    w3, h3 = CurrentMousePlace()
    print(w3, h3)
    w4, h4 = CurrentMousePlace()
    print(w4, h4)

    pyautogui.alert(text='准备开始', title='提示')

    MouseOperationDcl(w1, h1, time=0.5)
    time.sleep(1)
    MouseOperationDcl(w2, h2, time=0.5)
    time.sleep(1)
    MouseOperationCl(w3, h3, time=0.5)
    time.sleep(1)
    MouseOperationCl(w4, h4, time=0.5)
    time.sleep(1)

    for i in range(1, 1500):
        MouseOperationDcl(w1, h1, time=0.5)
        MouseOperationClPd(w2, h2, time=0.5)
        MouseOperationDcl(w2, h2, time=0.5)
        MouseOperationCl(w3, h3, time=0.5)
        time.sleep(1.5)
        MouseOperationCl(w4, h4, time=0.5)


if __name__ == '__main__':
    print(pyautogui.size())
    main()
