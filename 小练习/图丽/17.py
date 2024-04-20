import ctypes
import platform

# 加载 C 标准库
# 不同操作系统下，C 标准库的名称可能不同
if platform.system() == "Windows":
    libc = ctypes.CDLL("msvcrt.dll")
elif platform.system() in ["Linux", "Darwin"]:  # Linux 或 macOS
    libc = ctypes.CDLL(
        "libc.so.6" if platform.system() == "Linux" else "libSystem.dylib"
    )

# 定义 time 函数的原型
# time_t time(time_t *tloc);
# 在 ctypes 中，time_t 通常可以用 c_long 表示
libc.time.argtypes = [ctypes.POINTER(ctypes.c_long)]
libc.time.restype = ctypes.c_long

# 调用 time 函数
t = ctypes.c_long()
libc.time(ctypes.byref(t))

# 打印调用结果
print("Current time:", t.value)
