import sys
import traceback

__stderr__ = sys.stderr
sys.stderr = open("errorlog.txt", "w")

# 使用traceback函数定位错误信息
try:
    result = 1 / 0
except:
    traceback.print_exc()
