<<<<<<< HEAD
from scipy.io import loadmat

# Python正确理解的path中需采用\\与/作为分隔符号或采用r取消\的转义意义作为分隔符号
# ../ 表示当前文件所在的目录的上一级目录
# ./ 表示当前文件所在的目录(可以省略)
file = 'D:\\Study File\\MATLAB\\数据\\MyData.mat'
# file = r'D:\Study File\MATLAB\数据\MyData.mat'
# file = 'D:/Study File/MATLAB/数据/MyData.mat'
# 导入Mat数据
=======
# 导入Mat数据
from scipy.io import loadmat

# path中采用r取消转义或采用\\与/分隔目录级能使Python正确理解
# ../ 表示当前文件所在的目录的上一级目录
# ./ 表示当前文件所在的目录(可以省略)
file = 'D:\\Study File\\MATLAB\\数据\\MyData1.mat'
# file = r'D:\Study File\MATLAB\数据\MyData1.mat'
# file = 'D:/Study File/MATLAB/数据/MyData1.mat'
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000
# mat_dtype=True，保证了导入后变量的数据类型与原类型一致
data = loadmat(file, mat_dtype=True)
print(data['DisMat'].dtype)
print(data['DisMat'].size)