from sklearn import preprocessing
from scipy.stats import rankdata


x = [[1], [3], [34], [21], [10], [12]]
std_x = preprocessing.StandardScaler().fit_transform(x)
norm_x = preprocessing.MinMaxScaler().fit_transform(x)
print('原始顺序：', rankdata(x))
print('标准顺序：', rankdata(std_x))
print('归一顺序：', rankdata(norm_x))
