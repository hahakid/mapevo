import numpy as np


def Wasserstein(H, G):
    # 公式6 A Fast Histogram-Based Similarity Measure for Detecting Loop Closures in 3-D LIDAR Data
    l = H.shape[0]
    result = 0.0
    last = 0.0
    for i in range(0, l):
        last += abs(H[i] - G[i])
        result += last
    if l > 0:
        result /= l
    return result


# 公式 1-4 A Fast Histogram-Based Similarity Measure for Detecting Loop Closures in 3-D LIDAR Data
def Histogram_close_loop_dis(pc1, pc2, b, d_min, d_max):
    '''
    pc1, pc2 = ndarray[n*4]
    b = 切分粒度
    d_min = 最小距离范围
    d_max = 最大距离范围, 代码实现通过最大的idx<100 删掉这部分点，但是比较怀疑是不是有问题，因为最大点会产生波动
    delta_b = 为每个b切分后的区间长度
    1.
    遍历pc1 求每个点的三维距离d
    index=(d-d_min)/delta_b, 是否要丢弃一定距离d_max外的点，github找的这个算法没有实现。
    然后只保存了H=index[0,99]的值
    2.
    同样方法计算pc2，存入G
    3.
    归一化H和G
    4.
    返回，H和G
    '''

    H = np.zeros(b)
    G = np.zeros(b)
    l1 = pc1.shape[0]
    l2 = pc2.shape[0]
    delta_b = (d_max - d_min) / b

    for i in range(0, l1):
        d = np.sqrt(pc1[i, 0] ** 2 + pc1[i, 1] ** 2 + pc1[i, 2] ** 2)
        if d_min < d < d_max:
            idx = int((d - d_min) / delta_b)
            #   print(idx)
            if 0 <= idx < b:  # 构建 长度为100的直方图统计距离
                H[idx] += 1.0

    for i in range(0, l2):
        d = np.sqrt(pc2[i, 0] ** 2 + pc2[i, 1] ** 2 + pc2[i, 2] ** 2)
        if d_min < d < d_max:
            idx = int((d - d_min) / delta_b)
            if 0 <= idx < b:  # 构建 长度为100的直方图统计距离
                G[idx] += 1.0

    # normalization
    for i in range(0, b):
        if l1 > 0:
            H[i] /= l1
        if l2 > 0:
            G[i] /= l2

    # return H, G
    return Wasserstein(H, G)


# 公式 1-4 A Fast Histogram-Based Similarity Measure for Detecting Loop Closures in 3-D LIDAR Data
def Histogram_close_loop_height(pc1, pc2, b, h_min=0, h_max=10):
    '''
    pc1, pc2 = ndarray[n*4]
    b = 切分粒度
    h_min = 最低高度范围
    h_max = 最大高度范围
    delta_h = 为每个b切分后的区间长度
    1.
    遍历pc1 求每个点的三维距离d
    index=(d-d_min)/delta_b, 是否要丢弃一定距离d_max外的点，github找的这个算法没有实现。
    然后只保存了H=index[0,99]的值
    2.
    同样方法计算pc2，存入G
    3.
    归一化H和G
    4.
    返回，H和G
    '''

    H = np.zeros(b)
    G = np.zeros(b)
    l1 = pc1.shape[0]
    l2 = pc2.shape[0]
    delta_h = (h_max - h_min) / b

    for i in range(0, l1):
        # d = np.sqrt(pc1[i, 0] ** 2 + pc1[i, 1] ** 2 + pc1[i, 2] ** 2)
        height = pc1[i, 2] + 1.72 # 最简单的高度编码就是直接取z，即便传感器有一个固定安装高度，
        if h_min < height < h_max:
            idx = int((height - h_min) / delta_h)
            if 0 <= idx < b:  # 构建 长度为100的直方图统计距离
                H[idx] += 1.0

    for i in range(0, l2):
        # d = np.sqrt(pc2[i, 0] ** 2 + pc2[i, 1] ** 2 + pc2[i, 2] ** 2)
        height = pc2[i, 2] + 1.72
        if h_min < height < h_max:
            idx = int((height - h_min) / delta_h)
            if 0 <= idx < b:  # 构建 长度为100的直方图统计距离
                G[idx] += 1.0

    # normalization
    for i in range(0, b):
        if l1 > 0:
            H[i] /= l1
        if l2 > 0:
            G[i] /= l2

    # return H, G
    return Wasserstein(H, G)
