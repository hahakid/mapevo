# !/usr/bin/env python3
# coding=utf-8

import os
import sys

import array
# import cv2
import numpy as np
import numpy.linalg as LA
import open3d as o3d
import matplotlib.pyplot as plt
import transforms3d

from map_metrics.config import LidarConfig  # local data from lidar point cloud
import map_metrics.config as config
# 原版
#from map_metrics.metrics import mme, mpv, mom
#from map_metrics.utils.orthogonal import read_orthogonal_subset
# 修改版
from map_metrics.metrics_ import mme, mpv, mom, aggregate_map, mme_map, mpv_map, mom_map
#from map_metrics.utils.orthogonal_ import read_orthogonal_subset
import time
import segplane
import evo.main_rpe as main_rpe
import evo.main_ape as main_ape
from evo.core import metrics
from evo.core import trajectory

import pickle

# colormap
semantic_mapping = {  # bgr
    0: [0, 0, 0],  # "unlabeled", and others ignored
    1: [0, 0, 255],  # outliner
    10: [245, 150, 100],  # "floor"
    11: [245, 230, 100],  # "wall"
    12: [150, 60, 30],  # "pillar"
    13: [30, 30, 255],  # "celling"
    21: [180, 30, 80],  # "desk"
    22: [255, 0, 0],  # "chair"
}

def rigid_translate(pc_input, extrinsic):
    # projection
    pc = np.row_stack([pc_input[:, :3].T, np.ones_like(pc_input[:, 0])])
    pc = np.matmul(extrinsic, pc)
    pc = np.row_stack([pc[:3, :], pc_input[:, 3]]).T
    return pc

# 增加一个scale
def poseNoise(pose, mode, scale):
    """
    add noise to pose
    :param pose: 4*4 matrix
    :param  mode: 't', 'r' and 'rt'
    :param  scale: 1, 1.5, 2
    :return: pose with noise, and normalized error
    pose 变换 4*4 4*4，用矩阵乘法， pose（4*4） pc（n*3）是旋转 使用rigid_translate
    np.matmul(pose, TNoise) 的顺序先统一，不区分左右顺序
    t: m 米
    R: rad 弧
    """
    tNoiseHuge = 0.1 * scale  # meter
    tNoiseSmall = 0.005 * scale
    rNoiseHuge = 1 * np.pi / 180 * scale   # rad
    rNoiseSmall = 0.05 * np.pi / 180 * scale

    TNoise = np.eye(4)
    if mode == 't':
        x = np.random.uniform(-tNoiseHuge, tNoiseHuge)
        y = np.random.uniform(-tNoiseHuge, tNoiseHuge)
        z = np.random.uniform(-tNoiseSmall, tNoiseSmall)
        TNoise[:3, 3] = np.array([x, y, z]).reshape(-1)
        #print(pose, TNoise)
        pose = np.matmul(pose, TNoise)
        #print(pose)
        error = np.linalg.norm(TNoise[:3, 3])  # noise error of translation
        return pose, error
    if mode == 'r':
        roll = np.random.uniform(-rNoiseSmall, rNoiseSmall)
        pitch = np.random.uniform(-rNoiseSmall, rNoiseSmall)
        yaw = np.random.uniform(-rNoiseHuge, rNoiseHuge)
        TNoise[:3, :3] = transforms3d.euler.euler2mat(roll, pitch, yaw)
        #print(pose, TNoise)
        pose = np.matmul(pose, TNoise)  # rigid_translate(TNoise, pose)
        #print(pose)
        #pose = rigid_translate(TNoise, pose)
        # print(pose)
        error = np.linalg.norm(TNoise[:3, :3] - np.eye(3))
        return pose, error
    if mode == 'rt':
        x = np.random.uniform(-tNoiseHuge, tNoiseHuge)
        y = np.random.uniform(-tNoiseHuge, tNoiseHuge)
        z = np.random.uniform(-tNoiseSmall, tNoiseSmall)
        TNoise[:3, 3] = np.array([x, y, z]).reshape(-1)
        roll = np.random.uniform(-rNoiseSmall, rNoiseSmall)
        pitch = np.random.uniform(-rNoiseSmall, rNoiseSmall)
        yaw = np.random.uniform(-rNoiseHuge, rNoiseHuge)
        # print(roll, pitch, yaw)
        TNoise[:3, :3] = transforms3d.euler.euler2mat(roll, pitch, yaw)
        #print(pose, TNoise)
        pose = np.matmul(pose, TNoise)  # rigid_translate(TNoise, pose)
        #print(pose)
        #pose = rigid_translate(TNoise, pose)
        #print(pose)
        error = np.linalg.norm(TNoise - np.eye(4))
        return pose, error

# read labels
def load_label(label_file_path):
    label = np.fromfile(label_file_path, dtype=np.uint32)
    label = label.reshape((-1))
    return label

# get color map based on label idx
def getcolor(lab):
    color = []
    for i in lab:
        color.append(semantic_mapping[i])
    color = np.asarray(color) / 255.0
    return color

# load bin point cloud
def load_pc(bin_file_path):
    """
    load pointcloud file (velodyne format)
    :param bin_file_path:
    :return:
    """
    pc = np.fromfile(bin_file_path, dtype=np.float32).reshape((-1, 4))
    return pc

def load_kitti_odometry_poses(file):
    """
    :param file: The odometry file path
    :return: R, T matrix
    """
    poses_raw = np.loadtxt(file, dtype=np.float32)
    poses = [np.row_stack([p.reshape(3, 4), [0, 0, 0, 1]]) for p in poses_raw]
    return poses

def pc_show(pc, norm_flag=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)
    opt = vis.get_render_option()
    opt.point_size = 2
    opt.point_show_normal = norm_flag
    for p in pc:
        vis.add_geometry(p)
    vis.run()
    vis.destroy_window()

def npy2pcdColor(pcd_np, color_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(color_data[:, :3])
    return pcd

def npy2pcd(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    return pcd

def getline(tar, color):
    num_nodes = len(tar)
    lines = [[x, x + 1] for x in range(num_nodes - 1)]
    colors = np.tile(color, (len(lines), 1))
    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(tar)
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector(colors)
    return trajectory

def noiseTest():

    gtPoses = load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))

    assert len(gtPoses) == len(lidarFileList)

    winSize = 5  # 10, 15, 20, 25
    mapping = []
    color_all = []
    # begin = int(1+windowSize/2)-1, end = int(len(gtPoses)-windowSize/2)
    # remove the first and last ones.
    # winSize as step to downsample
    for i in range(int(1+winSize/2), int(len(gtPoses)-winSize/2), winSize):
        # print(i)
        #begin = int(1+windowSize/2)-1
        #end = int(len(gtPoses)-windowSize/2)+1
        curPose = gtPoses[i]
        estPose = poseNoise(curPose, 't', 1)  # gt pose with noise
        #print(curPose, estPose)
        #  to see what will the whole map be like after noise

        label_file = os.path.join(fatherPath, scene, 'labels', '{:06d}.label'.format(i))

        label = load_label(label_file)
        colors = getcolor(label)

        curPc = load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[i]))
        # curPc = rigid_translate(curPc, estPose)
        curPc = rigid_translate(curPc, curPose)

        mapping.append(curPc)
        color_all.append(colors)

    pc = np.row_stack(mapping)
    color_all = np.row_stack(color_all)
    pcd = npy2pcdColor(pc, color_all)
    pc_show([pcd])

def aggregateMap():
    '''
    show loacl aggregate map and trajectory
    both with noise and without noise can be selected.
    '''
    gtPoses = load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))
    assert len(gtPoses) == len(lidarFileList)
    redColor = np.array([1, 0, 0])
    greenColor = np.array([0, 1, 0])
    winSize = [5]  #, 10, 15, 20, 25]

    for w in winSize:
        step = w
        gtGlobalMap = []  # aggregate all the local GT point cloud
        estGlobalMap = []  # aggregate all the local EST point cloud
        #for i in range(int(1+w/2), int(len(gtPoses)-w/2), step):
        for i in range(113, int(len(gtPoses)-w/2-1), step):
            # print("i:", i)
            gtLocalMap = []  # aggregate current GT point cloud frames
            estLocalMap = []  # aggregate current EST point cloud frames
            gtTar = []
            estTar = []
            for j in range(int(i-w/2+1), int(i+w/2+1)):
                print(j)
                gtPose = gtPoses[j]  # get pose, the global coordinate
                gtCenter = gtPose[:3, 3]  # the t part
                # print(gtPose)

                estPose = poseNoise(gtPose, mode='t', scale=1)  # with noise
                estCenter = estPose[:3, 3]

                curPc = load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[j]))  # get point cloud
                gtPc = rigid_translate(curPc, gtPose)  # translate to global
                estPc = rigid_translate(curPc, estPose)  # with noise
                # save pc frames
                gtLocalMap.append(gtPc)
                estLocalMap.append(estPc)
                # save trajectories
                gtTar.append(gtCenter)
                estTar.append(estCenter)

            gtTrajectory = getline(gtTar, redColor)  # get open3d line set as red
            estTrajectory = getline(estTar, greenColor) # get open3d line set as green

            gtLocalAggMap = np.row_stack(gtLocalMap)
            estLocalAggMap = np.row_stack(estLocalMap)

            gtGlobalMap.append(gtLocalMap)
            estGlobalMap.append(estLocalMap)

            gtPcd = npy2pcd(gtLocalAggMap[:, :3])
            estPcd = npy2pcd(estLocalAggMap[:, :3])
            pc_show([estPcd, gtTrajectory, estTrajectory])  # visualize

# pose rpe
def localRpe(gtPoseSeq, estPoseSeq):
    '''
        delta = 1, delta_unit = frames
        full_transformation : unit-less
        translation_part : meters
        rotation_part : unit-less

        rotation_angle_rad : radians
        rotation_angle_deg : degrees
    '''
    # @ list of NDArray
    traj_ref = trajectory.PosePath3D(poses_se3=gtPoseSeq)  # 类型转化
    traj_est = trajectory.PosePath3D(poses_se3=estPoseSeq)  # 类型转化
    pose_relation_f = metrics.PoseRelation.full_transformation
    pose_relation_t = metrics.PoseRelation.translation_part
    pose_relation_r = metrics.PoseRelation.rotation_part  # 是否用单位呢？
    delta_unit = metrics.Unit.frames
    # delta = 1  # 对每帧都进行估计， 计算函数初始化之后会被强制类型转化为int
    data = (traj_ref, traj_est)
    # full
    rpe_metric1 = metrics.RPE(pose_relation=pose_relation_f, delta=1.0, delta_unit=delta_unit, all_pairs=False)
    rpe_metric1.process_data(data)
    rpe_stat1 = rpe_metric1.get_statistic(metrics.StatisticsType.rmse)  # rmse=sqrt(sse/n)
    # T
    rpe_metric2 = metrics.RPE(pose_relation=pose_relation_t, delta=1.0, delta_unit=delta_unit, all_pairs=False)
    rpe_metric2.process_data(data)
    rpe_stat2 = rpe_metric2.get_statistic(metrics.StatisticsType.rmse)  # rmse=sqrt(sse/n)
    # R
    rpe_metric3 = metrics.RPE(pose_relation=pose_relation_r, delta=1.0, delta_unit=delta_unit, all_pairs=False)
    rpe_metric3.process_data(data)
    rpe_stat3 = rpe_metric3.get_statistic(metrics.StatisticsType.rmse)  # rmse=sqrt(sse/n)
    #print("rpe full:", rpe_stat1)
    return rpe_stat1, rpe_stat2, rpe_stat3

# pose ape
def localApe(gtPoseSeq, estPoseSeq):
    traj_ref = trajectory.PosePath3D(poses_se3=gtPoseSeq)  # 类型转化
    traj_est = trajectory.PosePath3D(poses_se3=estPoseSeq)  # 类型转化
    pose_relation_f = metrics.PoseRelation.full_transformation
    pose_relation_t = metrics.PoseRelation.translation_part
    pose_relation_r = metrics.PoseRelation.rotation_part  # 是否用单位呢？
    delta_unit = metrics.Unit.frames
    data = (traj_ref, traj_est)

    # full
    rpe_metric1 = metrics.APE(pose_relation=pose_relation_f)
    rpe_metric1.process_data(data)
    rpe_stat1 = rpe_metric1.get_statistic(metrics.StatisticsType.rmse)  # rmse=sqrt(sse/n)
    # T
    rpe_metric2 = metrics.APE(pose_relation=pose_relation_t)
    rpe_metric2.process_data(data)
    rpe_stat2 = rpe_metric2.get_statistic(metrics.StatisticsType.rmse)  # rmse=sqrt(sse/n)
    # R
    rpe_metric3 = metrics.APE(pose_relation=pose_relation_r)
    rpe_metric3.process_data(data)
    rpe_stat3 = rpe_metric3.get_statistic(metrics.StatisticsType.rmse)  # rmse=sqrt(sse/n)

    return rpe_stat1, rpe_stat2, rpe_stat3


def wholeMap(fatherPath, scene):
    '''
    calculated the MOM, MPV, MME (map-based) and APE (full-ref) for a whole map
    and map-based need to be downsampled for huge computing
    首先计算full-ref，快一些。

    MOM计算：
    map: 原地图
    ds_map: 体素降采样地图
    基于ds_map 计算候选正交平面
    然后基于map的kd-tree进行局部搜索，求取 平面方差
    '''
    #  降采样会导致 knn搜索半径内没有合适的正交平面
    conf = config.CustomConfig(knn_rad=10.0, min_knn=10, max_nn=30, min_clust_size=10)  # use lidar config, or change with custom params
    downSampleMode = "voxel"
    gtPoses = load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))
    assert len(gtPoses) == len(lidarFileList)
    gtPcdSeq = []  # list of GT pcd, map is aggregated based on pose, so one pcd seq is needed
    gtPoseSeq = []  # list of GT pose tarjectory
    estPoseSeq = [] # list of EST pose tarjectory

    full = 0
    tPart = 0
    rPart = 0

    mmeList = []
    mpvList = []
    momList = []
    # some seq need to start at floor2:23; garage:72
    # for i in range(int(1+w/2-1), int(len(gtPoses)-w/2-1), step):
    for i in range(23, len(gtPoses)):  # fast debug
        gtPose = gtPoses[i]  # get pose, the global coordinate
        # gtCenter = gtPose[:3, 3]  # the t part
        gtPoseSeq.append(gtPose)  # save cur gt pose
        estPose = poseNoise(gtPose, 'rt', 1)  # with noise
        # estCenter = estPose[:3, 3]
        estPoseSeq.append(estPose)  # save cur est pose
        curPc = load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[i]))  # get point cloud
        pcd = npy2pcd(curPc)
        gtPcdSeq.append(pcd)

    full, tPart, rPart = localApe(gtPoseSeq, estPoseSeq)
    print("APE: full= %f T= %f  R= %f \n" % (full, tPart, rPart))

    gt_pc_map = aggregate_map(gtPcdSeq, gtPoseSeq)  # gt map: open3d pcd
    est_pc_map = aggregate_map(gtPcdSeq, estPoseSeq)  # est map: open3d pcd

    if downSampleMode == 'voxel':
        ds_step = [0.8, 0.5, 0.2, 0.1]  # voxel size (m)
        for ds in ds_step:
            print(ds)
            gt_ds_pc_map = gt_pc_map.voxel_down_sample(ds)
            # map-based metrics calculation
            Timer1 = time.time()
            #mmeList.append(mme_map(gt_ds_pc_map, conf))  # map lidar_conf
            Timer2 = time.time()
            #mpvList.append(mpv_map(gt_ds_pc_map, conf))  # map lidar_conf
            Timer3 = time.time()
            momList.append(mom_map(gt_pc_map, gt_ds_pc_map, None, conf))  # map ds_map None lidar_conf
            Timer4 = time.time()
            print("GT: downsample voxel ={%f}, time consume:\n mme: {%f}\n mpv: {%f}\n mom {%f}\n"
                  % (ds_step, Timer2-Timer1, Timer3-Timer2, Timer4-Timer3))

            est_ds_pc_map = est_pc_map.voxel_down_sample(ds)
            # map-based metrics calculation
            Timer1 = time.time()
            mmeList.append(mme_map(est_ds_pc_map, conf))  # map lidar_conf
            Timer2 = time.time()
            mpvList.append(mpv_map(est_ds_pc_map, conf))  # map lidar_conf
            Timer3 = time.time()
            momList.append(mom_map(est_pc_map, est_ds_pc_map, None, conf))  # map ds_map None lidar_conf
            Timer4 = time.time()
            print("EST: downsample voxel ={%f}, time consume:\n mme: {%f}\n mpv: {%f}\n mom {%f}\n"
                  % (ds_step, Timer2-Timer1, Timer3-Timer2, Timer4-Timer3))
    # 感觉这个是基于原始点云顺序进行采样，未考虑点云整体的拓扑，加上点云的随机性 可能破坏空间结构
    if downSampleMode == 'uniform':
        ds_step = [5, 10, 20, 50, 100]  # sample rate: every ds point
        for ds in ds_step:
            gt_ds_pc_map = gt_pc_map.uniform_down_sample(ds)  #
            est_ds_pc_map = est_pc_map.voxel_down_sample_and_trace(ds)

def windowSampling(fatherPath, scene):
    '''
    @ datapath
    @ scenename
    @ windowSize
    MOM, MPV, MME need list of pcd and pose as input
    calculated the MOM, MPV, MME (map-based) and RPE (full-ref) within each sliding window
    '''
    gtPoses = load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))
    assert len(gtPoses) == len(lidarFileList)
    # winSize = [5]  #, 10, 15, 20, 25]

    '''
    KNN_RAD = 1
    MIN_KNN = 5
    MAX_NN = 30
    MIN_CLUST_SIZE = 5
    '''
    conf = LidarConfig  # 默认雷达配置
    # conf = config.CustomConfig(knn_rad=1.0, min_knn=50, max_nn=200, min_clust_size=20)  # use lidar config, or change with custom params
    win = [5, 10, 15, 20, 25]
    for w in win:
        # map-based
        mmeList = []
        mpvList = []
        momList = []
        # trajectory-based
        rpeFullList = []
        rpeTList = []
        rpeRList = []
        step = w
        # some seq need to start at floor2:23; garage:72
        # for i in range(int(1+w/2-1), int(len(gtPoses)-w/2-1), step):
        for i in range(23, int(len(gtPoses)-w/2-1), step):  # fast debug
            # print("i:", i)
            '''
            pcd only need one sequence, because the noise are added on the pose,
            therefore, both the GT pose seq and EST pose seq are needed.
            '''
            gtPcdSeq = []  # list of GT pcd
            gtPoseSeq = []  # list of GT pose tarjectory
            estPoseSeq = [] # list of EST pose tarjectory
            for j in range(int(i-w/2+1), int(i+w/2+1)):  # a range with j in the middle
                # print(j)
                gtPose = gtPoses[j]  # get pose, the global coordinate
                # gtCenter = gtPose[:3, 3]  # the t part
                gtPoseSeq.append(gtPose)

                estPose, _ = poseNoise(gtPose, mode='rt', scale=1)  # with noise
                # estCenter = estPose[:3, 3]
                estPoseSeq.append(estPose)

                curPc = load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[j]))  # get point cloud
                pcd = npy2pcd(curPc)
                gtPcdSeq.append(pcd)

            # map-based metrics
            #'''
            Timer1 = time.time()
            #mmeList.append(mme(gtPcdSeq, estPoseSeq, conf))
            Timer2 = time.time()
            #mpvList.append(mpv(gtPcdSeq, estPoseSeq, conf))
            Timer3 = time.time()
            momList.append(mom(gtPcdSeq, estPoseSeq, None, conf))
            Timer4 = time.time()
            print("time consume:\n mme: {%f}\n mpv: {%f}\n mom {%f}\n" % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3))
            #'''
            # evo rpe
            full, tPart, rPart = localRpe(gtPoseSeq, estPoseSeq)
            rpeFullList.append(full)
            rpeTList.append(tPart)
            rpeRList.append(rPart)
        # 以时间窗口为单位进行输出
        print("window:", w, mmeList, mpvList, momList, rpeFullList, rpeTList, rpeRList)


def windowSamplingMulitPorcess(fatherPath, scene, win, noisemode, scale, outFileName):
    '''
    @ datapath
    @ scenename
    @ windowSize
    MOM, MPV, MME need list of pcd and pose as input
    calculated the MOM, MPV, MME (map-based) and RPE (full-ref) within each sliding window
    '''
    gtPoses = load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))
    assert len(gtPoses) == len(lidarFileList)
    # winSize = [5]  #, 10, 15, 20, 25]

    '''
    KNN_RAD = 1
    MIN_KNN = 5
    MAX_NN = 30
    MIN_CLUST_SIZE = 5
    '''
    conf = LidarConfig  # 默认雷达配置
    # conf = config.CustomConfig(knn_rad=1.0, min_knn=50, max_nn=200, min_clust_size=20)  # use lidar config, or change with custom params

    # map-based
    mmeList = []
    mpvList = []
    momList = []
    # trajectory-based
    rpeFullList = []
    rpeTList = []
    rpeRList = []
    errList = []

    PlaneMomList = []  # SPV
    normalEntropyList = []  # PNE
    planeVariacneList = []  # CPV
    planeNormalsList = []  # CPV

    # some seq need to start at floor2:23; garage:72
    # for i in range(int(1+w/2-1), int(len(gtPoses)-w/2-1), step):
    for i in range(23, int(len(gtPoses)-win/2-1), win):  # fast debug
        print("i:", i)
        '''
        pcd only need one sequence, because the noise are added on the pose,
        therefore, both the GT pose seq and EST pose seq are needed.
        '''
        gtPcdSeq = []  # list of GT pcd in current window
        gtGlobalPcdSeq = []  # list of GT pcd in current window in global coordinate
        gtPoseSeq = []  # list of GT pose tarjectory in current window
        estPoseSeq = [] # list of EST pose tarjectory in current window
        errSeq = []  # list of the normalized errors in current window

        colorSeq = []
        labelSeq = []

        candPlanes = []  # all planes in local map
        candNormals = []  # the normals of the planes in local map

        orthPlanes = []  # list of orthogonal planes in frame(j)
        for j in range(int(i-win/2+1), int(i+win/2+1)):  # a range with j in the middle
            # print(j)
            gtPose = gtPoses[j]  # get pose, the global coordinate
            # gtCenter = gtPose[:3, 3]  # the t part
            gtPoseSeq.append(gtPose)

            estPose, curError = poseNoise(gtPose, noisemode, scale)  # with noise
            # estCenter = estPose[:3, 3]
            estPoseSeq.append(estPose)

            curPc = load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[j]))  # get point cloud
            pcd = npy2pcd(curPc)
            gtPcdSeq.append(pcd)
            errSeq.append(curError)  # save errors

            # Translate the curPc into global coordinate for plane based metric
            globalCurPc = rigid_translate(curPc, estPose)
            gtGlobalPcdSeq.append(globalCurPc)

            # curlabel = load_label(os.path.join(fatherPath, scene, 'labels', '{:06d}.label'.format(j)))  # gt
            curlabel = load_label(os.path.join(fatherPath, scene, 'predictions', '{:06d}.label'.format(j)))  # best model
            labelSeq = labelSeq + list(curlabel)  # label (1D) in cur window

            colors = getcolor(curlabel)
            colorSeq.append(colors)  # color in cur window
            curPlanes, curNormals = segplane.segPlane(globalCurPc, colors, curlabel)  # j帧=[pc, color, label]

            if j == i:  # 只对中间帧求候选正交平面
                orthPlanes = segplane.find_max_clique(curPlanes, curNormals)  #

            candPlanes += curPlanes  # list<pcd>
            candNormals += curNormals  # list<ndarray 1*3>

        errList.append(sum(errSeq)/len(errSeq))  # mean of the normalized errors in current window = mNE?
        # map-based metrics
        #'''
        Timer1 = time.time()
        mmeList.append(mme(gtPcdSeq, estPoseSeq, conf))
        Timer2 = time.time()
        mpvList.append(mpv(gtPcdSeq, estPoseSeq, conf))
        Timer3 = time.time()
        momList.append(mom(gtPcdSeq, estPoseSeq, None, conf))
        Timer4 = time.time()
        #print("time consume:\n mme: {%f}\n mpv: {%f}\n mom {%f}\n" % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3))
        #'''


        # evo rpe
        full, tPart, rPart = localRpe(gtPoseSeq, estPoseSeq)
        rpeFullList.append(full)
        rpeTList.append(tPart)
        rpeRList.append(rPart)

        Timer5 = time.time()
        # plane based mom
        # 基于当前窗口的中间帧，中存在的候选正交平面，在当前窗口进行求解
        PlaneMomList.append(segplane.localOrthPlaneMetric(gtGlobalPcdSeq, colorSeq, orthPlanes))

        Timer6 = time.time()
        # plane normals entropy, PNE
        nEntropy = segplane.normalClusteringEntropy(candNormals)  #
        normalEntropyList.append(nEntropy)  # normal entropy for r
        Timer7 = time.time()
        # parallel plane variance, PPV
        pVariance, pNormals = segplane.parallePlane(candPlanes, candNormals)  # plane variance for t
        Timer8 = time.time()
        planeNormalsList.append(pNormals)
        planeVariacneList.append(pVariance)

        #print(Timer8-Timer7)
        with open(outFileName, 'a') as sfile:
            sfile.writelines("time consume:\n mme: %f\n mpv: %f\n mom %f\n SPV: %f\n PNE: %f\n CPV+CPN: %f\n"
                             % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3, Timer6-Timer5, Timer7-Timer6, Timer8-Timer7))

    # 以时间窗口为单位进行输出
    with open(outFileName, 'a') as sfile:
    #print("window:", win, mmeList, mpvList, momList, rpeFullList, rpeTList, rpeRList)
        sfile.writelines("window: %d\n" % (win))

        sfile.writelines("noise: \n")
        sfile.writelines("GT Noise: "+str(errList)+"\n")
        sfile.writelines("map-based metrics:\n")
        sfile.writelines("MME: "+str(mmeList)+"\n")
        sfile.writelines("MPV: "+str(mpvList)+"\n")
        sfile.writelines("MOM: "+str(momList)+"\n")
        sfile.writelines("reference-based metrics:\n")
        sfile.writelines("RPE-Full: "+str(rpeFullList)+"\n")
        sfile.writelines("RPE-T: "+str(rpeTList)+"\n")
        sfile.writelines("RPE-R: "+str(rpeRList)+"\n")
        sfile.writelines("new-plane-based metrics:\n")
        sfile.writelines("Plane-MOM: "+str(PlaneMomList)+"\n")
        sfile.writelines("PNE: "+str(normalEntropyList)+"\n")
        sfile.writelines("CPV: "+str(planeVariacneList)+"\n")
        sfile.writelines("CPN: "+str(planeNormalsList)+"\n")

def windowSamplingMulitPorcessCurrent(fatherPath, scene, win, noisemode, scale, outFileName):
    '''
    @ datapath
    @ scenename
    @ windowSize
    MOM, MPV, MME need list of pcd and pose as input
    calculated the MOM, MPV, MME (map-based) and RPE (full-ref) within each sliding window
    '''
    gtPoses = load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))
    assert len(gtPoses) == len(lidarFileList)
    # winSize = [5]  #, 10, 15, 20, 25]

    '''
    KNN_RAD = 1
    MIN_KNN = 5
    MAX_NN = 30
    MIN_CLUST_SIZE = 5
    '''
    conf = LidarConfig  # 默认雷达配置
    # conf = config.CustomConfig(knn_rad=1.0, min_knn=50, max_nn=200, min_clust_size=20)  # use lidar config, or change with custom params

    # map-based
    mmeList = []
    mpvList = []
    momList = []
    # trajectory-based
    rpeFullList = []
    rpeTList = []
    rpeRList = []
    errList = []

    PlaneMomList = []  # SPV
    normalEntropyList = []  # PNE
    planeVariacneList = []  # CPV
    planeNormalsList = []  # CPV

    # some seq need to start at floor2:23; garage:72
    # for i in range(int(1+w/2-1), int(len(gtPoses)-w/2-1), step):
    for i in range(23, int(len(gtPoses)-win/2-1), win):  # fast debug
        print("i:", i)
        '''
        pcd only need one sequence, because the noise are added on the pose,
        therefore, both the GT pose seq and EST pose seq are needed.
        '''
        gtPcdSeq = []  # list of GT pcd in current window
        gtGlobalPcdSeq = []  # list of GT pcd in current window in global coordinate
        gtPoseSeq = []  # list of GT pose tarjectory in current window
        estPoseSeq = [] # list of EST pose tarjectory in current window
        errSeq = []  # list of the normalized errors in current window

        colorSeq = []
        labelSeq = []

        candPlanes = []  # all planes in local map
        candNormals = []  # the normals of the planes in local map

        orthPlanes = []  # list of orthogonal planes in frame(j)
        for j in range(int(i-win/2+1), int(i+win/2+1)):  # a range with j in the middle
            #print(j)
            gtPose = gtPoses[j]  # get pose, the global coordinate
            # gtCenter = gtPose[:3, 3]  # the t part
            gtPoseSeq.append(gtPose)

            if i == j:
                estPose, curError = poseNoise(gtPose, noisemode, scale)  # with noise
                # estCenter = estPose[:3, 3]
                estPoseSeq.append(estPose)
            else:
                curError = np.linalg.norm(np.eye(4) - np.eye(4))
                estPoseSeq.append(gtPose)

            curPc = load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[j]))  # get point cloud
            pcd = npy2pcd(curPc)
            gtPcdSeq.append(pcd)
            errSeq.append(curError)  # save errors

            # Translate the curPc into global coordinate for plane based metric
            if i == j:
                globalCurPc = rigid_translate(curPc, estPose)
            else:
                globalCurPc = rigid_translate(curPc, gtPose)

            gtGlobalPcdSeq.append(globalCurPc)
            # curlabel = load_label(os.path.join(fatherPath, scene, 'labels', '{:06d}.label'.format(j)))  # gt
            curlabel = load_label(os.path.join(fatherPath, scene, 'predictions', '{:06d}.label'.format(j)))  # best model
            labelSeq = labelSeq + list(curlabel)  # label (1D) in cur window

            colors = getcolor(curlabel)
            colorSeq.append(colors)  # color in cur window
            curPlanes, curNormals = segplane.segPlane(globalCurPc, colors, curlabel)  # j帧=[pc, color, label]

            if j == i:  # 只对中间帧求候选正交平面
                orthPlanes = segplane.find_max_clique(curPlanes, curNormals)  #

            candPlanes += curPlanes  # list<pcd>
            candNormals += curNormals  # list<ndarray 1*3>

        errList.append(sum(errSeq)/len(errSeq))  # mean of the normalized errors in current window = mNE?
        # map-based metrics
        #'''
        Timer1 = time.time()
        mmeList.append(mme(gtPcdSeq, estPoseSeq, conf))
        Timer2 = time.time()
        mpvList.append(mpv(gtPcdSeq, estPoseSeq, conf))
        Timer3 = time.time()
        momList.append(mom(gtPcdSeq, estPoseSeq, None, conf))
        Timer4 = time.time()
        # print("time consume:\n mme: {%f}\n mpv: {%f}\n mom {%f}\n" % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3))
        #'''


        # evo rpe
        full, tPart, rPart = localRpe(gtPoseSeq, estPoseSeq)
        rpeFullList.append(full)
        rpeTList.append(tPart)
        rpeRList.append(rPart)

        Timer5 = time.time()
        # plane based mom
        # 基于当前窗口的中间帧，中存在的候选正交平面，在当前窗口进行求解
        PlaneMomList.append(segplane.localOrthPlaneMetric(gtGlobalPcdSeq, colorSeq, orthPlanes))

        Timer6 = time.time()
        # plane normals entropy, PNE
        nEntropy = segplane.normalClusteringEntropy(candNormals)  #
        normalEntropyList.append(nEntropy)  # normal entropy for r
        Timer7 = time.time()
        # parallel plane variance, PPV
        pVariance, pNormals = segplane.parallePlane(candPlanes, candNormals)  # plane variance for t
        Timer8 = time.time()
        planeNormalsList.append(pNormals)
        planeVariacneList.append(pVariance)

        #print(Timer8-Timer7)
        with open(outFileName, 'a') as sfile:
            sfile.writelines("time consume:\n mme: %f\n mpv: %f\n mom %f\n SPV: %f\n PNE: %f\n CPV+CPN: %f\n"
                             % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3, Timer6-Timer5, Timer7-Timer6, Timer8-Timer7))

    # 以时间窗口为单位进行输出
    with open(outFileName, 'a') as sfile:
        #print("window:", win, mmeList, mpvList, momList, rpeFullList, rpeTList, rpeRList)
        sfile.writelines("window: %d\n" % (win))

        sfile.writelines("noise: \n")
        sfile.writelines("GT Noise: "+str(errList)+"\n")
        sfile.writelines("map-based metrics:\n")
        sfile.writelines("MME: "+str(mmeList)+"\n")
        sfile.writelines("MPV: "+str(mpvList)+"\n")
        sfile.writelines("MOM: "+str(momList)+"\n")
        sfile.writelines("reference-based metrics:\n")
        sfile.writelines("RPE-Full: "+str(rpeFullList)+"\n")
        sfile.writelines("RPE-T: "+str(rpeTList)+"\n")
        sfile.writelines("RPE-R: "+str(rpeRList)+"\n")
        sfile.writelines("new-plane-based metrics:\n")
        sfile.writelines("Plane-MOM: "+str(PlaneMomList)+"\n")
        sfile.writelines("PNE: "+str(normalEntropyList)+"\n")
        sfile.writelines("CPV: "+str(planeVariacneList)+"\n")
        sfile.writelines("CPN: "+str(planeNormalsList)+"\n")

from multiprocessing import Pool

#def multi_run_wrapper_windowSampling(args):
#    return windowSampling(*args)

if __name__ == '__main__':
    fatherPath = r'J:\map\savedframes'
    scene = 'floor2' #'office1' #
    #noiseTest()
    #aggregateMap()
    # multi-porcess with diff windowSize
    win = 5
    noisemode = 'rt'
    scale = 1
    outFileName = "./results/win=%d mode=%s scale=%.1f.txt" % (win, noisemode, scale)
    #windowSamplingMulitPorcess(fatherPath, scene, win, noisemode, scale, outFileName)
    windowSamplingMulitPorcessCurrent(fatherPath, scene, win, noisemode, scale, outFileName)
    '''
    arg_list = []
    #for winSize in range(5, 16, 5):
    for winSize in range(3, 8, 2):
        for mode in ['t', 'r', 'rt']:
            for scale in [1, 1.5, 2]:
                print(winSize, mode, scale)
                outFileName = "win=%d mode=%s scale=%.1f.txt" % (winSize, mode, scale)
                outFileName = os.path.join('./results', outFileName)
                arg_list.append((fatherPath, scene, winSize, mode, scale, outFileName))
                #print(outFileName)
    # print(arg_list)
    # arg_list = [(fatherPath, scene, 5), (fatherPath, scene, 10), (fatherPath, scene, 15), (fatherPath, scene, 20), (fatherPath, scene, 25)]
    with Pool() as pool:
        pool.starmap(func=windowSamplingMulitPorcess, iterable=arg_list)  # rpe and local_map
    '''
    #wholeMap(fatherPath, scene)  # ape and global_map


