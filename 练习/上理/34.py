import sys
import traceback

__stderr__ = sys.stderr
sys.stderr = open('errorlog.txt', 'w')

# 使用traceback函数定位错误信息
try:
<<<<<<< HEAD
    result = 1 / 0
=======
    1 / 0
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000
except:
    traceback.print_exc()