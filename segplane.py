# !/usr/bin/env python3
# coding=utf-8

import os
import sys
import networkx as nx

import array
# import cv2
import numpy as np
import numpy.linalg as LA
import open3d as o3d
import matplotlib.pyplot as plt
import transforms3d
from scipy.stats import entropy
from map_metrics.config import LidarConfig  # local data from lidar point cloud
from map_metrics.metrics_ import _plane_variance, _entropy
import map_metrics.config as config
# 原版
#from map_metrics.metrics import mme, mpv, mom
#from map_metrics.utils.orthogonal import read_orthogonal_subset
# 修改版
from map_metrics.metrics_ import mme, mpv, mom, aggregate_map, mme_map, mpv_map, mom_map
#from map_metrics.utils.orthogonal_ import read_orthogonal_subset
import time

#import evo.main_rpe as main_rpe
#import evo.main_ape as main_ape
#from evo.core import metrics
#from evo.core import trajectory

import pickle
import reader
from sklearn.cluster import AgglomerativeClustering


def visPlaneBox(boxes, pcd):
    geometries = []
    for obox in boxes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])  # flatten bbox along normal
        #mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox)
        mesh.paint_uniform_color(obox.color)
        geometries.append(mesh)
        geometries.append(obox)
    geometries.append(pcd)  # add full pcd
    return geometries

def ransacPlaneSeg(pcd):
    planeList = []
    normalList = []
    while np.asarray(pcd.points).shape[0] > 30:
        # print(np.asarray(pcd.points).shape[0], len(planeList))
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=100)
        # print(plane_model)
        if len(inliers) >= 50:  # remove too small planes
            normals = plane_model[:3]
            inlier_cloud = pcd.select_by_index(inliers)
            # 需要先估计法向，不然法向为空
            inlier_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=50))
            inlier_cloud.orient_normals_to_align_with_direction(normals)  # reset points with plane normal
            #print(np.asarray(inlier_cloud.normals))
            #inlier_cloud.normalize_normals()  # 法向归一化
            #print(np.asarray(inlier_cloud.normals))
            # inlier_cloud.paint_uniform_color(normals)  # with color
            reader.pc_show([inlier_cloud])
            planeList.append(inlier_cloud)
            normalList.append(normals)
        # 剩余点
        pcd = pcd.select_by_index(inliers, invert=True)
    return planeList, normalList

def robustStatisticPlaneSeg(pcd):
    planeList = []
    normalList = []
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=50))
    #reader.pc_show([perClassPcd], norm_flag=True)  # 查看normal
    # 只执行一次
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0, #0.3,   #0,
        min_num_points=0, #50,   #0,
        #search_param=o3d.geometry.KDTreeSearchParamRadius(radius=1))  # different search scheme
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    geometries = []
    for box in oboxes:
        curClassPcd = pcd.crop(box)  # crop based on box
        # refresh the normals based on plane
        # normal = box.R[:, 2]
        #print("1:", np.asarray(curClassPcd.normals))
        if np.asarray(curClassPcd.points).shape[0] > 50:  # filtering smale group planes
            curClassPcd.orient_normals_to_align_with_direction(box.R[:, 2])  # reset normal based on plane
            #print("2:", np.asarray(curClassPcd.normals))
            # curClassPcd.normalize_normals()  # 法向归一化， 没有影响 开始应该就计算了
            # print("3:", np.asarray(curClassPcd.normals))
            planeList.append(curClassPcd)
            normalList.append(box.R[:, 2])  # 如果上面已经重置了normal，是否还有必要存储normalList值得考虑，另外就是检测边缘其实还有些异常值
        ''' visualization

        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(box.color)
        geometries.append(mesh)
        geometries.append(curClassPcd)
        reader.pc_show(geometries, norm_flag=True)
        '''
    return planeList, normalList

# 似乎没多少变化，除非修改第一部分的平面拟合
def rubustStatisticPlaneSegIter(pcd):
    planeList = []
    normalList = []
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=50))
    # 考虑基于第一轮估计的normal重新再迭代一次平面估计
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0.3,   #0,
        min_num_points=50,   #0,
        #search_param=o3d.geometry.KDTreeSearchParamRadius(radius=1))  # different search scheme
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    geometries = []
    pcd_combined = o3d.geometry.PointCloud()
    for box in oboxes:
        curClassPcd = pcd.crop(box)  # crop based on box
        curClassPcd.orient_normals_to_align_with_direction(box.R[:, 2])  # reset normal based on plane
        pcd_combined = pcd_combined + curClassPcd
        planeList.append(curClassPcd)
        #normalList.append(box.R[:, 2])
        #''' visualization
        #geometries = []
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(box.color)
        geometries.append(mesh)
        geometries.append(curClassPcd)
        reader.pc_show(geometries, norm_flag=True)
        #'''
    geometries = []
    oboxes = pcd_combined.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0.3,   #0,
        min_num_points=50,   #0,
        #search_param=o3d.geometry.KDTreeSearchParamRadius(radius=1))  # different search scheme
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))

    for box in oboxes:
        curClassPcd = pcd_combined.crop(box)  # crop based on box
        curClassPcd.orient_normals_to_align_with_direction(box.R[:, 2])  # reset normal based on plane
        planeList.append(curClassPcd)
        normalList.append(box.R[:, 2])
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(box.color)
        geometries.append(mesh)
        geometries.append(curClassPcd)

    reader.pc_show(geometries, norm_flag=True)
    return planeList, normalList


def windowSampling(fatherPath, scene, win, noisemode, scale, outFileName):

    gtPoses = reader.load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))  # pose list
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))  # pc list
    labelFileList = os.listdir(os.path.join(fatherPath, scene, 'labels'))  # label list
    assert len(gtPoses) == len(lidarFileList) == len(labelFileList)
    conf = LidarConfig  # 默认雷达配置

    # map-based
    mmeList = []
    mpvList = []
    momList = []
    # trajectory-based
    rpeFullList = []
    rpeTList = []
    rpeRList = []
    errList = []
    # some seq need to start at floor2:23; garage:72
    for i in range(327, int(len(gtPoses)-win/2-1), win):  # fast debug
        print("i:", i)
        gtPcdSeq = []  # list of GT pcd in current window
        gtPoseSeq = []  # list of GT pose tarjectory in current window
        estPoseSeq = [] # list of EST pose tarjectory in current window
        errSeq = [] # list of the normalized errors in current window
        colorSeq = []
        labelSeq = []
        candidatePlanes = []  # All detected planes in current window
        candidateNormals = []  # the related normals of the detected planes in current window
        for j in range(j, int(i+win/2+1)):#int(i-win/2+1), int(i+win/2+1)):  # a range with j in the middle
            # print(j)
            gtPose = gtPoses[j]  # get pose, the global coordinate
            gtPoseSeq.append(gtPose)

            estPose, curError = reader.poseNoise(gtPose, noisemode, scale)  # with noise
            # estCenter = estPose[:3, 3]
            estPoseSeq.append(estPose)

            curPc = reader.load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[j]))  # get point cloud

            curPc = reader.rigid_translate(curPc, estPose)
            gtPcdSeq.append(curPc)  # pc in cur window
            # 读取分割结果时，替换文件目录
            curlabel = reader.load_label(os.path.join(fatherPath, scene, 'labels', '{:06d}.label'.format(j)))
            labelSeq = labelSeq + list(curlabel)  # label (1D) in cur window

            colors = reader.getcolor(curlabel)
            colorSeq.append(colors)  # color in cur window
            # print("pc:", curPc.shape, " color:", colors.shape, " label:", curlabel.shape)
            #gtPcdSeq.append(pcd)
            errSeq.append(curError)  # save errors

        #print(len(gtPcdSeq), len(colorSeq), np.row_stack(labelSeq).reshape(-1))
            # local map within current window
            pcTotal = np.row_stack(gtPcdSeq)
            colorTotal = np.row_stack(colorSeq)
            labelIndex = np.unique(labelSeq)  # unique label index for
            allPcd = []
            for lIdx in labelIndex:
                if lIdx >= 10 and lIdx <= 20:  # 0/1= noise and unlabeled; >20, sim-stable object
                    #print("label:", lIdx)
                    ind = np.where(labelSeq == lIdx)
                    perClassPc = pcTotal[ind]
                    perClassColor = colorTotal[ind]
                    perClassPcd = reader.npy2pcdColor(perClassPc, perClassColor)
                    #reader.pc_show([perClassPcd])
                    # add plane seg on perClassPcd
                    # subPcd, subNormal = localPlaneSeg(perClassPcd)
                    subPcd, subNormal = robustStatisticPlaneSeg(perClassPcd)
                    #subPcd, subNormal = rubustStatisticPlaneSegIter(perClassPcd)
                    allPcd = allPcd + subPcd  # combine = list + list
                    #print(len(subPcd))
                    #reader.pc_show(subPcd)
                    # 先估计法向(实际上我觉得如果进行了发现估计，边缘点可能会出问题，可视化以下看看)

            reader.pc_show(allPcd, norm_flag=True)
                    # for seg test
                    #name = os.path.join('./pcd_4_seg', 'frame_%d_cls_%d' % (j, lIdx) + '.pcd')
                    #o3d.io.write_point_cloud(name, perClassPcd)

        errList.append(sum(errSeq)/len(errSeq))  # mean of the normalized errors in current window = mNE?

        #reader.pc_show(gtPcdSeq)


#
def find_max_clique(candPlanes, candNormals, eps=1e-1):
    '''
    :param candPlanes:  List<o3d.pcd>
    :param candNormals: List<ndarray>
    :param eps: 0.1
    :return: List<o3d.pcd>
    '''
    N = len(candNormals)
    adj_matrix = np.zeros((N, N))
    #  基于法向垂直，构建邻接矩阵
    #  将当前簇与剩余簇将进行遍历，相当于只遍历对角阵，但是最后2句都填充了，是个对称矩阵，主对角线为0
    #   基于共线和正交关系，增加关联
    for i in range(N):  #
        for j in range(i):  #
            x = np.abs(np.dot(candNormals[i], candNormals[j]))  # |n_i dot n_j|
            if x < eps:  # nearly co-line， 可能是同一平面， 可视化观测时，会发现同一平面上的点的法向正好相反
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
            if x > 1-eps:  # nearly orthogonal， 正交平面
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    D = nx.Graph(adj_matrix)  # 基于邻接矩阵构建图
    x = nx.algorithms.clique.find_cliques(D)  # 搜索团

    full_cliques_size = []
    full_cliques = []
    for clique in x:
        if len(clique) > 2:  # 规模大于2的团，每个团里面应该包含多个聚类簇
            amount = 0
            for j in clique:  # 遍历团内的每个簇， 获得团包含的点数量
                amount += np.asarray(candPlanes[j].points).shape[0]  # 累加：对应 cluster 编号的点
            full_cliques_size.append(amount)  # 团的数量
            full_cliques.append(clique)  #

    if len(full_cliques) == 0:
        #raise ValueError("Length of full_cliques == 0")
        return None
    # 找到其中最大的一个团，输出
    max_ind = full_cliques_size.index(max(full_cliques_size))
    orthPlanes = []
    # orthNormal = []
    for i in full_cliques[max_ind]:
        orthPlanes.append(candPlanes[i])
        # orthNormal += candNormals[i]
    return orthPlanes #, orthNormal


def segPlane(pcd, color, label):
    '''
    :param pcd: ndarray n*3
    :param color: ndarray n*3
    :param label: list n
    :return:
    '''
    candidatePlanes = []  # All detected planes
    candidateNormals = []
    labelIndex = np.unique(label)
    for lIdx in labelIndex:  # filter by seg index
        if lIdx >= 10 and lIdx <= 20:  # 0/1= noise and unlabeled; >20, sim-stable object
            #print("label:", lIdx)
            ind = np.where(label == lIdx)
            #print(ind, pcd.shape)
            perClassPc = pcd[ind]
            perClassColor = color[ind]
            perClassPcd = reader.npy2pcdColor(perClassPc, perClassColor)
            #reader.pc_show([perClassPcd])
            # add plane seg on perClassPcd
            # subPcd, subNormal = ransacPlaneSeg(perClassPcd)
            subPcd, subNormal = robustStatisticPlaneSeg(perClassPcd)  # o3d.pcd, ndarray
            candidatePlanes += subPcd  # o3d.pcd list combine use +
            candidateNormals += subNormal  # ndarray list combine use +
            #allPcd = allPcd + subPcd  # combine = list + list
    #reader.pc_show(candidatePlanes, norm_flag=True)
    return candidatePlanes, candidateNormals


def localOrthPlaneMetric(gtPcdSeq, colorSeq, orthPlanes):
    #PlaneMomList = []
    pcTotal = np.row_stack(gtPcdSeq)
    colorTotal = np.row_stack(colorSeq)
    # labelIndex = np.unique(labelSeq)  # unique label index, may useless in following calculation
    # print(pcTotal.shape, len(labelSeq))
    #add 基于pcTotal 构建 kdtree, 遍历orthPlanes进行局部搜索，进行平面方差估计

    CurWinMap = reader.npy2pcdColor(pcTotal, colorTotal)
    map_tree = o3d.geometry.KDTreeFlann(CurWinMap)  # full local map tree
    mapPoints = np.asarray(CurWinMap.points)
    orth_axes_stats = []
    # search the local planes based on orthPlanes on pcs
    for planes in orthPlanes:  # iter each planes
        metric = []
        # add the farthest distance down sampling, to reduce computation at 1/10
        # all points are too density
        size = np.asarray(planes.points).shape[0]
        planes_down = planes.farthest_point_down_sample(int(size/10))  # down-sample
        # planes_down = planes
        for p in range(np.asarray(planes_down.points).shape[0]):  # iter each points in current plane
            point = np.asarray(planes_down.points)[p]
            _, idx, _ = map_tree.search_radius_vector_3d(point, radius=1)
            if len(idx) > 10:
                metric.append(_plane_variance(mapPoints[idx]))
                #print(len(idx))
        avg_metric = np.median(metric)  # 中值？
        orth_axes_stats.append(avg_metric)

    localMapMPV = np.sum(orth_axes_stats)  # 所有平面的均值和
    #PlaneMomList.append(localMapMPV)
    return localMapMPV

# scheme 0, calculated candidate orthogonal planes based on the middle frame of current window
def orthPlaneMidFrame(fatherPath, scene, win, noisemode, scale, outFileName):
    '''
    基于中间帧进行平面分割，计算正交平面候选集，然后基于正交平面候选集上的点进行遍历
    遍历前，基于每个面进行1/10规模降采样的 最大距离搜索，
    在整个窗口内所有帧的点云构建的KD-tree进行局部平面搜索
    计算 平面方差均值 作为结果
    :param fatherPath:
    :param scene:
    :param win:
    :param noisemode:
    :param scale:
    :param outFileName:
    :return:
    '''
    # print(outFileName)
    gtPoses = reader.load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))  # pose list
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))  # pc list
    labelFileList = os.listdir(os.path.join(fatherPath, scene, 'labels'))  # label list
    assert len(gtPoses) == len(lidarFileList) == len(labelFileList)
    conf = LidarConfig  # 默认雷达配置

    # map-based
    mmeList = []
    mpvList = []
    momList = []


    # trajectory-based
    rpeFullList = []
    rpeTList = []
    rpeRList = []
    errList = []
    # some seq need to start at floor2:23; garage:72
    for i in range(23, int(len(gtPoses)-win/2-1), win):  # fast debug
        print("i:", i)  #, #outFileName)
        gtPcdSeq = []  # list of GT pcd in current window
        gtPoseSeq = []  # list of GT pose tarjectory in current window
        estPoseSeq = [] # list of EST pose tarjectory in current window
        errSeq = [] # list of the normalized errors in current window
        colorSeq = []
        labelSeq = []


        orthPlanes = []  # detected orth planes in current window (from mid frame)
        for j in range(int(i-win/2+1), int(i+win/2+1)):  # a range with j in the middle
            # print(j)
            gtPose = gtPoses[j]  # get pose, the global coordinate
            gtPoseSeq.append(gtPose)

            estPose, curError = reader.poseNoise(gtPose, noisemode, scale)  # with noise
            # estCenter = estPose[:3, 3]
            estPoseSeq.append(estPose)

            curPc = reader.load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[j]))  # get point cloud

            curPc = reader.rigid_translate(curPc, estPose)  # local frame to global
            gtPcdSeq.append(curPc)  # pc in cur window
            # 读取分割结果时，替换文件目录
            curlabel = reader.load_label(os.path.join(fatherPath, scene, 'labels', '{:06d}.label'.format(j)))
            labelSeq = labelSeq + list(curlabel)  # label (1D) in cur window

            colors = reader.getcolor(curlabel)
            colorSeq.append(colors)  # color in cur window
            # print("pc:", curPc.shape, " color:", colors.shape, " label:", curlabel.shape)
            #gtPcdSeq.append(pcd)
            errSeq.append(curError)  # save errors
            if j == i:
                candPlanes, candNormals = segPlane(curPc, colors, curlabel)
                orthPlanes = find_max_clique(candPlanes, candNormals)  #

            #print(len(gtPcdSeq), len(colorSeq), np.row_stack(labelSeq).reshape(-1))
            # local map within current window
            if orthPlanes == None:
                print(win, noisemode, scale, i)
    PlaneMomList = localOrthPlaneMetric(gtPcdSeq, colorSeq, orthPlanes)

    with open(outFileName, 'a') as sfile:
        sfile.writelines(str(PlaneMomList)+"\n")

def _Entropy1(data):
    '''
    :param dataList: N*3 ndarray or
    :return: entropy according to
    A three-dimensional point cloud registration based on entropy and particle swarm optimization
    '''

    data_norm = np.linalg.norm(data, axis=0, keepdims=True)
    data = data / data_norm
    entropy_sum = 0
    for d in range(data.shape[1]):
        ent = entropy(data[:, d], base=np.e)
        if ent > 0:
            entropy_sum += ent
    #ent_x = entropy(data[:, 0], base=np.e)
    #ent_y = entropy(data[:, 1], base=np.e)
    #ent_z = entropy(data[:, 2], base=np.e)
    return entropy_sum

def _Entropy(NorList):
    cov = np.cov(NorList.T)
    det = np.linalg.det(2 * np.pi * np.e * cov)
    if det > 0:
        return 0.5 * np.log(det)
    return None

def normalClusteringEntropy(nlist):
    '''
    这里不能直接对所有法向直接计算，相当于没有分组，因此熵铁定是十分混乱的
    因为整个场景有太多不同法向的平面，直接算就是随机了
    参考基于法向的聚类，然后对每个簇单独计算法向的熵，丢弃规模较小的簇，最后基于簇求均值，作为最终结果
    :param nlist:
    :return:
    '''
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-1, compute_full_tree=True
    ).fit(nlist)
    # perClusterNormalEntropy = []
    labels = clustering.labels_
    n_clusters = np.unique(labels).shape[0]
    metric = []
    nnlist = np.asarray(nlist)
    # EntropyList = []
    for i in range(n_clusters):
        ind = np.where(labels == i)

        if ind[0].shape[0] > 3:
            # print(ind[0].shape[0])
            curCluster = nnlist[ind]
            metric_value = _Entropy1(curCluster)
            if metric_value is not None:
                metric.append(metric_value)
    # aaa = np.asarray(nlist)

    return 0.0 if len(metric) == 0 else np.mean(metric)

def parallelPlaneDistTest(i, j, P1, P2, N1, N2, eps):
    [a1, b1, c1, d1], _ = P1.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
    [a2, b2, c2, d2], _ = P2.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)
    dist1 = np.abs(d1 - d2) / np.sqrt(a1**2 + b1**2 + c1**2)
    #dist2 = np.abs(d1 - d2) / np.sqrt(a2**2 + b2**2 + c2**2)
    #print(i, j, dist1, dist2)

    '''
    print(i, j, [a1, b1, c1, d1], [a2, b2, c2, d2])
    print(i, j, N1, N2)
    print("est:", i, j, abs(a1/a2-b1/b2), abs(b1/b2-c1/c2), abs(a1/a2-c1/c2))
    print("normals:", i, j, abs(N1[0]/N2[0]-N1[1]/N2[1]), abs(N1[1]/N2[1]-N1[2]/N2[2]), abs(N1[2]/N2[2]-N1[0]/N2[0]))
    if np.abs(a1/a2 - b1/b2) < eps and np.abs(b1/b2 - c1/c2) < eps and np.abs(c1/c2 - a1/a2) < eps:
        dist = np.abs(d1 - d2) / np.sqrt(a1**2 + b1**2 + c1**2)
        return dist
    else:
        return np.inf
    '''
    return dist1

def parallelPlaneDist(p1, p2, eps):
    # normals 没有 d, 重新拟合一下
    [a1, b1, c1, d1], _ = p1.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=30)
    [a2, b2, c2, d2], _ = p2.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=30)
    dist1 = np.abs(d1 - d2) / np.sqrt(a1**2 + b1**2 + c1**2)
    dist2 = np.abs(d1 - d2) / np.sqrt(a2**2 + b2**2 + c2**2)
    if dist1 < eps * 2 and abs(dist1-dist2) < eps * 0.5:
        return dist1
    else:
        return np.inf

# coplane eva
def parallePlane(planes, normals, eps=1e-1):
    # reader.pc_show(planes)
    ''' 部分平面在估计后，总体的符号会发生变化，但是误差值基本在小数点后4位，把这部分逻辑挪到后面
    # get plane model first [A, B, C, D]
    planeModleList = []
    for p in planes:
        plane_model, _ = p.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=30)
        planeModleList.append(plane_model)

    x1 = np.asarray(planeModleList)
    x2 = np.asarray(normals)
    '''

    #b1 = time.time()
    #find clique
    N = len(planes)
    adj_matrix = np.zeros((N, N))

    for i in range(N):  #
        for j in range(i):  #
            x = np.abs(np.dot(normals[i], normals[j]))  # |n_i dot n_j|
            if x > 1 - eps:  # filtering parallel planes
                # planes[i].paint_uniform_color([1, 0, 0])
                # planes[j].paint_uniform_color([0, 0, 1])
                # reader.pc_show([planes[i], planes[j]])
                dist = parallelPlaneDist(planes[i], planes[j], eps)
                #reader.pc_show([planes[i], planes[j]])
                if dist != np.inf:  # 法向共线，且距离小于阈值,则添加边，需要调节阈值，可能有点苛刻，放缩eps倍数？
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

    D = nx.Graph(adj_matrix)  # 基于邻接矩阵构建图
    x = nx.algorithms.clique.find_cliques(D)  # 搜索团

    full_cliques = []  # 排查了完全没有冗余，都是不相交的子集
    for clique in x: #
        if len(clique) > 2:  # 规模大于2的团，每个团里面应该包含多个聚类簇
            full_cliques.append(clique)  #
    # print(full_cliques)
    # 基于每个团 计算平面方差
    #b2 = time.time()

    #b1 = time.time()
    planeVarianceList = []
    for c in full_cliques:
        coplane = o3d.geometry.PointCloud()
        for p in c:
            coplane += planes[p]
        pv = _plane_variance(np.asarray(coplane.points))
        planeVarianceList.append(pv)
    pv_metric = np.median(planeVarianceList)
    #b3 = time.time()
    planeNormalsList = []
    for c in full_cliques:
        normalList = []
        for p in c:
            normalList.append(normals[p])
        planeNormalsList.append(normalClusteringEntropy(normalList))
    planeNormalsList = np.asarray(planeNormalsList, dtype=np.float32)
    #print(planeNormalsList)
    planeNormalsList = planeNormalsList[planeNormalsList != np.array(None)]
    #print(planeNormalsList)
    pn_metric = np.mean(planeNormalsList)
    #b4 = time.time()
    #print(b2-b1, b3-b2, b4-b3)
    return pv_metric, pn_metric

# scheme 1, calculated candidate orthogonal planes for all frames in current window
def orthPlaneMidFrame1(fatherPath, scene, win, noisemode, scale, outFileName):
    '''
    基于窗口内所有帧进行平面分割，作为总的候选集合
    1. 基于所有分割平面的法向，求熵，评估法向波动 （针对r）
    2. 基于 平行条件、平面距离，构建共面平面子集
        基于每个共面平面子集 估计平面方差
        求取所有子集平面方差的均值 （针对t）
    :param fatherPath:
    :param scene:
    :param win:
    :param noisemode:
    :param scale:
    :param outFileName:
    :return:
    '''
    # print(outFileName)
    gtPoses = reader.load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))  # pose list
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))  # pc list
    labelFileList = os.listdir(os.path.join(fatherPath, scene, 'labels'))  # label list
    assert len(gtPoses) == len(lidarFileList) == len(labelFileList)
    conf = LidarConfig  # 默认雷达配置

    # map-based
    mmeList = []
    mpvList = []
    momList = []


    # trajectory-based
    rpeFullList = []
    rpeTList = []
    rpeRList = []
    errList = []

    normalEntropyList = []
    planeVariacneList = []
    # some seq need to start at floor2:23; garage:72
    for i in range(23, int(len(gtPoses)-win/2-1), win):  # fast debug
        # print("i:", i)  #, #outFileName)
        gtPcdSeq = []  # list of GT pcd in current window
        gtPoseSeq = []  # list of GT pose tarjectory in current window
        estPoseSeq = [] # list of EST pose tarjectory in current window
        errSeq = [] # list of the normalized errors in current window
        colorSeq = []
        labelSeq = []


        # orthPlanes = []  # detected orth planes in current window (from mid frame)

        candPlanes = []  # all planes in local map
        candNormals = []  # the normals of the planes in local map

        for j in range(int(i-win/2+1), int(i+win/2+1)):  # a range with j in the middle
            # print(j)
            gtPose = gtPoses[j]  # get pose, the global coordinate
            gtPoseSeq.append(gtPose)

            estPose, curError = reader.poseNoise(gtPose, noisemode, scale)  # with noise
            # estCenter = estPose[:3, 3]
            estPoseSeq.append(estPose)

            curPc = reader.load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[j]))  # get point cloud

            curPc = reader.rigid_translate(curPc, estPose)  # local frame to global
            gtPcdSeq.append(curPc)  # pc in cur window
            # 读取分割结果时，替换文件目录
            curlabel = reader.load_label(os.path.join(fatherPath, scene, 'labels', '{:06d}.label'.format(j)))
            labelSeq = labelSeq + list(curlabel)  # label (1D) in cur window

            colors = reader.getcolor(curlabel)
            colorSeq.append(colors)  # color in cur window
            # print("pc:", curPc.shape, " color:", colors.shape, " label:", curlabel.shape)
            #gtPcdSeq.append(pcd)
            errSeq.append(curError)  # save errors
            #if j == i:
            curPlanes, curNormals = segPlane(curPc, colors, curlabel)
            # orthPlanes = find_max_clique(candPlanes, candNormals)  #

            #print(len(gtPcdSeq), len(colorSeq), np.row_stack(labelSeq).reshape(-1))
            # local map within current window
            #if orthPlanes == None:
            #    print(win, noisemode, scale, i)
            candPlanes += curPlanes  # list<pcd>
            candNormals += curNormals  # list<ndarray 1*3>
        # print(len(candPlanes), len(candNormals))
        nEntropy = normalClusteringEntropy(candNormals)  #
        normalEntropyList.append(nEntropy)  # normal entropy for r
        pVariance, pNormals = parallePlane(candPlanes, candNormals)  # plane variance for t
        planeVariacneList.append(pVariance)
    #PlaneMomList = localOrthPlaneMetric(gtPcdSeq, colorSeq, orthPlanes)

    #with open(outFileName, 'a') as sfile:
    #    sfile.writelines(str(PlaneMomList)+"\n")
    #print(normalEntropyList, len(normalEntropyList))
    #print(planeVariacneList, len(planeVariacneList))
from multiprocessing import Pool


if __name__ == '__main__':
    fatherPath = r'J:\map\savedframes'
    scene = 'floor2' #'garage' #
    arg_list = []
    win = 10
    noisemode = 'rt'
    scale = 2
    outFileName = "./results/Plane_win=%d mode=%s scale=%.1f.txt" % (win, noisemode, scale)
    #windowSampling(fatherPath, scene, win=15, noisemode='rt', scale=2, outFileName='outFileName')
    #orthPlaneMidFrame(fatherPath, scene, win, noisemode, scale, outFileName)
    orthPlaneMidFrame1(fatherPath, scene, win, noisemode, scale, outFileName)
    '''
    for winSize in range(5, 16, 5):
        for mode in ['t', 'r', 'rt']:
            for scale in [1, 1.5, 2]:
                outFileName = "Plane_win=%d mode=%s scale=%.1f.txt" % (winSize, mode, scale)
                # print(outFileName)
                outFileName = os.path.join('./results', outFileName)
                arg_list.append((fatherPath, scene, winSize, mode, scale, outFileName))            
    with Pool() as pool:
        pool.starmap(func=orthPlaneMidFrame, iterable=arg_list)  # rpe and local_map
    '''
