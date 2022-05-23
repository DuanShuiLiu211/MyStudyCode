# 必须部分「接口配置」
from distutils.core import setup, Extension
from Cython.Build import cythonize
# 可选部分「用到的库」
import numpy as np  


extensions = [
    Extension(
    name='DotCython',
    sources=['./DotCython.pyx'],
    language = 'c',
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]),
    ]


setup(
    name='DotCython',
    ext_modules=cythonize(extensions
))

# name 是我们要生成的动态链接库的名字
# sources 里面可以包含 .pyx 文件，以及后面如果我们要调用 C/C++ 程序的话，还可以往里面加 .c / .cpp 文件
# language 其实默认就是 c，如果要用 c++，就改成 c++ 
# include_dirs 这个就是传给 gcc 的 -I 参数，源码中调用 numpy 所以添加参数
# library_dirs 这个就是传给 gcc 的 -L 参数
# libraries 这个就是传给 gcc 的 -l 参数
# extra_compile_args 就是传给 gcc 的额外的编译参数，比方说你可以传一个 -std=c++11
# extra_link_args 就是传给 gcc 的额外的链接参数