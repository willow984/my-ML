import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    # 加载数据
    fr = open(fileName)
    dataSet = []
    for line in fr.readlines():                               # 对每一行：
        curLine = line.strip().split('\t')                    # 去头尾空白，以tab分割，返回的是每一行的字符列表
        fltLine = [float(item) for item in curLine]           # 对curLine中每个元素用float()，
        dataSet.append(fltLine)
    return dataSet

def distEclud(vecA, vecB):
    # 计算欧氏距离
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    # 在数据范围内找k个随机的质心
    n = np.shape(dataSet)[1]                                  # 列数(维度)
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]                                  # 样本点数量
    clusterAssment = np.mat(np.zeros((m, 2)))                 # m行2列，用来记录每个样本点当前数据哪个簇，以及和该簇的最近距离
    centroids = createCent(dataSet, k)                        # k行n列记录每个簇质心位置的向量，先随机生成了K个质心
    clusterChanged = True                                     # 记录样本点所属的簇是否发生了变化，每一次遍历中任何样本点所属簇发生了变化说明还没到最优
    while clusterChanged:
        clusterChanged = False
        for i in range(m):      # 对每个样本点
            minDist = np.inf
            minIndex = -1
            for j in range(k):  # 每个样本点i要和所有质心点j算一次距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:    # 如果样本点跟某一个质心的距离更小
                    minDist = distJI    # 更新最小距离
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True                         # 有样本所属簇发生了变化，还需要继续遍历
            clusterAssment[i, :] = minIndex, minDist**2       # 记下该样本最新的所属簇和欧氏距离
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]     # 分别摘出属于k类的数据
            centroids[cent, :] = np.mean(ptsInClust, axis=0)                        # 计算当前簇所有点的平均值作为质心
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    '''
    m: 样本数量
    clusterAssment: m*2矩阵，记录每个样本所属簇以及在该簇中距离，形如[簇, 欧式距离]
    centroid0: 初始中心点
    centList: 一维列表，记录所有簇中心点的坐标
    '''
    # 二分k-means
    m = np.shape(dataSet)[0]   # 获取行数(样本数)
    clusterAssment = np.mat(np.zeros((m,2)))  # m*2矩阵，记录每个样本所属簇以及在该簇中距离
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]   # 找第一个中心点：计算数据样本中每一列的均值，返回所有点的均值作为初始中心点(np.mean返回二维数组)
    centList = [centroid0]     # 记录簇中心点坐标(初始化1个，即centroid0)
    for j in range(m):         # 对所有的样本
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2     # 更新clusterAssment中每一个样本点的欧氏距离
    while (len(centList) < k):  # 中心点数量少于k个时
        lowestSSE = np.inf      # 初始化最低的总平方误差为无穷大
        for i in range(len(centList)):  # 遍历所有的簇中心点,对当前每一个簇要在做一次二分kmeans
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]  # 取该簇中所有样本点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 在这个簇中使用kMeans算法再二分聚类，返回再聚类后的[样本点所属簇,距离]，[簇，质心坐标]
            sseSplit = sum(splitClustAss[:,1])  # 划分后的SSE
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])   # 划分前的SSE
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:    # 如果当前分割的总平方误差小于目前记录的最低误差，更新分割方案
                bestCentToSplit = i                     # 记录分割效果最好的那个簇
                bestNewCents = centroidMat              # 二分后两个新的质心
                bestClustAss = splitClustAss.copy()     # 复制当前分割的聚类分配情况[]
                lowestSSE = sseSplit + sseNotSplit      # 更新最低总平方误差
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)    # 将分割出来的其中一个新簇的分配编号更新为质心列表的长度
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit  # 将另一个新簇的分配编号设为原先要被分割的质心的编号
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]   # 更新簇中心列表，用一个新的簇中心替换原来的簇中心
        centList.append(bestNewCents[1,:].tolist()[0])              # 将另一个新的簇中心加入到簇中心列表中
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss    # 更新 clusterAssment 矩阵，反映新的簇分配情况
    return np.mat(centList), clusterAssment    # 返回最终的簇中心列表和样本点的簇分配情况

def distSLC(vecA, vecB):
    a = np.sin(vecA[0,1]*np.pi/180) * np.sin(vecB[0,1]*np.pi/180)
    b = np.cos(vecA[0,1]*np.pi/180) * np.cos(vecB[0,1]*np.pi/180) * \
                      np.cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
    return np.arccos(a + b)*6371.0

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    clusterClubs(3)


