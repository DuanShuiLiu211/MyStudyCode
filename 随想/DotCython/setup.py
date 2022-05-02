# 必须部分
from distutils.core import setup, Extension
from Cython.Build import cythonize
# 可选部分「用到的库」
import numpy as np

# 调用numpy所以添加include_dirs参数，没有则可以去掉
ext_modules = [Extension("DotCython", ["DotCython.pyx"], include_dirs=[np.get_include()]), ]
setup(ext_modules=cythonize(Extension(
    'DotCython',
    sources=['DotCython.pyx'],
    language='c',
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))