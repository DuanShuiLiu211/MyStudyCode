import time

from PIL import ImageGrab


def z(f):
    def y(*x):
        start_time = time.time()
        f(*x)  # 被包装的对象
        stop_time = time.time()
        delta_time = stop_time - start_time
        return print("{:.9f}".format(delta_time))

    return y


@z
def python_capture(filename, *place):
    screen = ImageGrab.grab()
    print("显示分辨率:{}".format(screen.size))
    img = ImageGrab.grab(bbox=place)
    img.save(filename)

    python_capture("./2.png", 0, 0, 1980, 1024)
