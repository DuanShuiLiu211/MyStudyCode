from scipy.io import loadmat

# Python正确理解的path中需采用\\与/作为分隔符号或采用r取消\的转义意义作为分隔符号
# ../ 表示当前文件所在的目录的上一级目录
# ./ 表示当前文件所在的目录(可以省略)
file = 'D:\\Study File\\MATLAB\\数据\\MyData.mat'
# file = r'D:\Study File\MATLAB\数据\MyData.mat'
# file = 'D:/Study File/MATLAB/数据/MyData.mat'
# 导入Mat数据
# mat_dtype=True，保证了导入后变量的数据类型与原类型一致
data = loadmat(file, mat_dtype=True)
print(data['DisMat'].dtype)
print(data['DisMat'].size)