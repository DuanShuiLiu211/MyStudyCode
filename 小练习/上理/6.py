import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.io import loadmat
import time


# 计算n个欧几里得距离返回n*n大小的距离矩阵
def compute_squared_EDM(X):
    return squareform(pdist(X, metric='euclidean'))


# DBSCAN核心算法
def DBSCAN(data, eps, minPts):
    # 获取n个数据距离矩阵
    disMat = compute_squared_EDM(data)
    # 获得数据的行和列(n 数据量，m 数据维度)
    n, m = data.shape
    # 将矩阵的中小于eps的数赋予1，大于eps的数赋予零，axis=1代表对每一行求和,然后求核心点坐标的索引
    core_points_index = np.where(np.sum(np.where(disMat <= eps, 1, 0), axis=1) >= minPts)[0]
    # 初始化类别，-1代表未分类。
    labels = np.full((n,), -1)
    clusterId = 0
    # 遍历所有的核心点
    for pointId in core_points_index:
        # 如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
        if (labels[pointId] == -1):
            # 首先将点pointId标记为当前类别(即标识为已操作)
            labels[pointId] = clusterId
            # 然后寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
            neighbour = np.where((disMat[:, pointId] <= eps) & (labels == -1))[0]
            seeds = set(neighbour)
            # 通过种子点，开始生长，寻找密度可达的数据点，一直到种子集合为空，一个簇集寻找完毕
            while len(seeds) > 0:
                # 弹出一个新种子点
                newPoint = seeds.pop()
                # 将newPoint标记为当前类
                labels[newPoint] = clusterId
                # 寻找newPoint种子点eps邻域（包含自己）
                queryResults = np.where(disMat[:, newPoint] <= eps)[0]
                # 如果newPoint属于核心点，那么newPoint是可以扩展的，即密度是可以通过newPoint继续密度可达的
                if len(queryResults) >= minPts:
                    # 将邻域内且没有被分类的点压入种子集合
                    for resultPoint in queryResults:
                        if labels[resultPoint] == -1:
                            seeds.add(resultPoint)
            # 簇集生长完毕，寻找到一个类别
            clusterId = clusterId + 1
    return labels


# 将分类后的数据可视化显示
<<<<<<< HEAD
def plot_feature(data, labels):
=======
def plotFeature(data, labels):
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000
    clusterNum = len(set(labels))
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(-1, clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels == i)]
        ax.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=12, alpha=0.6)
    plt.show()
<<<<<<< HEAD
    

# 加载数据
data, _ = datasets.make_moons(500, noise=0.1, random_state=1)
plt.scatter(data[:, 0], data[:, 1], s=6, alpha=0.6)

# DBSCAN聚类并返回标识，eps=0.25，且MinPts=12
start = time.time()
labels = DBSCAN(data, 0.25, 12)
end = time.time()
print('Time Cost: %s' % str(end - start))

# 可视化
plot_feature(data, labels)
=======


# 加载数据
# data, _ = datasets.make_moons(500, noise=0.1, random_state=1)
file = 'W:\Study Flies\MTALAB\Data\MyData1.mat'
data1 = loadmat(file, mat_dtype=True)
data = data1['Test']
print(data)
start = time.time()
# DBSCAN聚类并返回标识,eps=0.25，且MinPts=12
labels = DBSCAN(data, 10, 12)
end = time.time()
print('Time Cost: %s' % str(end - start))
plotFeature(data, labels)
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000
