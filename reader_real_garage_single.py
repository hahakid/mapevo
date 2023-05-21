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

from multiprocessing import Pool

# colormap
semantic_mapping = {  # bgr
    0: [0, 0, 0],  # "unlabeled", and others ignored # filtered
    1: [0, 0, 255],  # outliner # filtered
    10: [245, 150, 100],  # "floor"
    11: [245, 230, 100],  # "wall"
    12: [150, 60, 30],  # "pillar"
    13: [30, 30, 255],  # "celling"  # filtered
    21: [180, 30, 80],  # "vehicle" # filtered
    22: [255, 0, 0],  # "person"    # filtered
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

def poseErr(gt, est):
    return abs(np.linalg.norm(gt - est))


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

# pose rpe, @ two sequences of pose
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
        for i in range(0, int(len(gtPoses)-w/2-1), step):  # fast debug
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
        '''
        Timer1 = time.time()
        mmeList.append(mme(gtPcdSeq, estPoseSeq, conf))
        Timer2 = time.time()
        mpvList.append(mpv(gtPcdSeq, estPoseSeq, conf))
        Timer3 = time.time()
        momList.append(mom(gtPcdSeq, estPoseSeq, None, conf))
        Timer4 = time.time()
        #print("time consume:\n mme: {%f}\n mpv: {%f}\n mom {%f}\n" % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3))
        '''


        # evo rpe
        full, tPart, rPart = localRpe(gtPoseSeq, estPoseSeq)
        rpeFullList.append(full)
        rpeTList.append(tPart)
        rpeRList.append(rPart)

        Timer5 = time.time()
        # plane based mom
        # 基于当前窗口的中间帧，中存在的候选正交平面，在当前窗口进行求解
        # PlaneMomList.append(segplane.localOrthPlaneMetric(gtGlobalPcdSeq, colorSeq, orthPlanes))

        Timer6 = time.time()
        # plane normals entropy, PNE
        PNE = segplane.normalClusteringEntropy(candNormals)  # PNE
        normalEntropyList.append(PNE)  # normal entropy for r
        Timer7 = time.time()
        # parallel plane variance, PPV
        #pVariance, pNormals = segplane.parallePlane(candPlanes, candNormals)  #
        CPV = segplane.parallePlane1(candPlanes, candNormals)  #

        Timer8 = time.time()
        planeNormalsList.append(CPV)
        #planeVariacneList.append(pVariance)

        #print(Timer8-Timer7)
        #with open(outFileName, 'a') as sfile:
        #    sfile.writelines("time consume:\n mme: %f\n mpv: %f\n mom %f\n SPV: %f\n PNE: %f\n CPV+CPN: %f\n"
        #                     % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3, Timer6-Timer5, Timer7-Timer6, Timer8-Timer7))

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
        #sfile.writelines("Plane-MOM: "+str(PlaneMomList)+"\n")
        sfile.writelines("PNE: "+str(normalEntropyList)+"\n")
        #sfile.writelines("CPV: "+str(planeVariacneList)+"\n")
        sfile.writelines("CPN: "+str(planeNormalsList)+"\n")

def normalized(list):
    arr = np.asarray(list)
    norm = np.linalg.norm(arr)
    arr = arr / norm
    return arr.tolist()

def windowSamplingMulitPorcessCurrent(fatherPath, scene, outFileName):
    '''
    @ datapath
    @ scenename
    @ windowSize
    MOM, MPV, MME need list of pcd and pose as input
    calculated the MOM, MPV, MME (map-based) and RPE (full-ref) within each sliding window

    在之前的计算过程中，我们只考虑了添加噪声对地图评估带来的影响，但是没有考虑GT情况下（默认了GT是不会有任何影响的）
    但是，如果考察GT和EST 在新的实验中打算加入新的柱状图（est+abs(gt-est)），来体现相关性的变化（不一定有效，因为两个并不一定正相关）。
    这是昨天跑步的时候想到的。

    除了RPE和curError相关指标不需要区分，其余其实均需要进行区分存储
    '''
    # output after interactive-slam based on estPoses
    gtPoses = load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses.txt'))
    # output of SLAM algorithm
    estPoses = load_kitti_odometry_poses(file=os.path.join(fatherPath, scene, 'poses1.txt'))
    lidarFileList = os.listdir(os.path.join(fatherPath, scene, 'velodyne'))
    assert len(gtPoses) == len(lidarFileList) == len(estPoses)
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
    mmeListGT = []
    mpvListGT = []
    momListGT = []

    # map1-based
    mmeListEST = []
    mpvListEST = []
    momListEST = []

    # trajectory-based
    rpeFullList = []
    rpeTList = []
    rpeRList = []
    errList = []

    PlaneMomList = []  # SPV
    normalEntropyList = []  # PNE
    planeVariacneList = []  # CPV
    planeNormalsList = []  # CPV


    # 存坐标
    coordList = []
    # real garage start from 0,
    for i in range(106, len(gtPoses) - 2):
    #for i in range(0, int(len(gtPoses)-win/2-1), win):  # fast debug
        print("i:", i)
        '''
        pcd only need one sequence, because the noise are added on the pose,
        therefore, both the GT pose seq and EST pose seq are needed.
        '''

        PcdSeq = []  # list of pcd in current window
        gtGlobalPcdSeq = []  # list of GT (pcd * gtPose) in current window in global coordinate
        estGlobalPcdSeq = []  # list of GT (pcd * estPose) in current window in global coordinate

        gtPoseSeq = []  # list of GT pose in current window
        estPoseSeq = [] # list of EST pose in current window

        errSeq = []  # list of the normalized errors in current window

        colorSeq = []
        #   labelSeq = []

        candPlanes = []  # all planes in local map
        candNormals = []  # the normals of the planes in local map

        orthPlanes = []  # list of orthogonal planes in frame(j)
        '''
        对RPE的计算而言，需要至少连续2帧的gt和est，|gt(t)-gt(t+1)| 与 |est(t)-est(t+1)| 之间的差异
        对map-based, 也是需要至少连续两帧点云一起来计算地图的拓扑差异，但是这时候同样存在，gt-map 和 est-map (分别基于两个pose叠加出来的点云地图进行计算)
        那么这个时候引出两个问题：1. RPE是单独的，我们只是求相关性，之前是求扭曲之后的，并没有考虑与gt-pose叠出来地的差异。
        2.也没有考虑将t和R的分量拆出来求相关性（因为实际噪声必然已经包含了两个分量的噪声）
        这时候从上面两个角度增加实验看看？
        '''

        for j in range(i, i+2):  # a range with j in the middle
            # print(j)
            #直接基于连续2帧进行计算，不再使用窗口概念。rpe需要 gt(i,i+1) 和 est(i,i+1)
            # interactive-slam pose
            gtPose = gtPoses[j]  # get pose, the global coordinate
            gtPoseSeq.append(gtPose)
            # gtCenter = gtPose[:3, 3]  # the t part
            # 只保存当前点坐标，用于绘制2D轨迹图
            if i == j:
                # aaa = tuple(gtPose[:3, 3].T)
                coordList.append(tuple(gtPose[:3, 3]))
            # slam pose
            estPose = estPoses[j]
            estPoseSeq.append(estPose)

            # error 之前主要用 np.linalg.norm(TNoise - np.eye(4))，然后基于窗口 # 后来发现跟 求均值无关，单帧没有窗口的概念了
            # error 用于计算三元相关性 [poseError, Map-based, RPE]
            # 确实不记得当时的计算逻辑，似乎就是看噪声本身有多大，现在需要基于 norm(estPose-gtPose)
            if i == j:
                curError = poseErr(gtPose, estPose)
                errSeq.append(curError)

            # pcd
            curPc = load_pc(os.path.join(fatherPath, scene, 'velodyne', lidarFileList[j]))  # get point cloud
            pcd = npy2pcd(curPc)
            PcdSeq.append(pcd)  # only local pcd without pose


            # pcd to global coord based on gt pose
            gtglobalCurPc = rigid_translate(curPc, gtPose)  # current frame
            # pcd to global coord based on est pose
            estglobalCurPc = rigid_translate(curPc, estPose)  # current frame


            gtGlobalPcdSeq.append(gtglobalCurPc)  # list of 2-continuous frames
            estGlobalPcdSeq.append(estglobalCurPc)  # list of 2-continuous frames

            curlabel = load_label(os.path.join(fatherPath, scene, 'labels', '{:06d}.label'.format(j)))  # gt labeled by human
            # curlabel = load_label(os.path.join(fatherPath, scene, 'predictions', '{:06d}.label'.format(j)))  # best model
            # labelSeq = labelSeq + list(curlabel)  # label (1D) in cur window

            colors = getcolor(curlabel)
            colorSeq.append(colors)  # color in cur window

            # 对当前连续帧平面进行语义分割，并过滤动态语义目标，基于colormap着色

            curPlanes, curNormals = segplane.segPlane(gtglobalCurPc, colors, curlabel)  # j帧=[pc, color, label]

            # print(len(curPlanes))
            # 求当前帧正交平面候选集
            # orthPlanes = segplane.find_max_clique(curPlanes, curNormals)  # 计算每帧的正交平面候选集，然后并集后一起计算后续

            candPlanes += curPlanes  # list<pcd>
            candNormals += curNormals  # list<ndarray 1*3>



        # evo rpe, 不变，计算只与pose相关
        full, tPart, rPart = localRpe(gtPoseSeq, estPoseSeq)
        rpeFullList.append(full)
        rpeTList.append(tPart)
        rpeRList.append(rPart)

        #errList.append(sum(errSeq)/len(errSeq))  # mean of the normalized errors in current window = mNE
        errList += errSeq  # 单帧时没有这个概念了，直接存入 norm(gt_pose - est_pose)
        #print(len(PcdSeq))
        # map-based metrics with noise
        Timer1 = time.time()
        #print('mme')
        #mmeListEST.append(mme(PcdSeq, estPoseSeq, conf))  # 基于交互优化前的估计，相当于噪声严重
        #mmeListGT.append(mme(PcdSeq, gtPoseSeq, conf))  # 基于交互优化后的估计，相当于降噪之后的
        Timer2 = time.time()
        #print('mpv')
        #mpvListEST.append(mpv(PcdSeq, estPoseSeq, conf))  #
        #mpvListGT.append(mpv(PcdSeq, gtPoseSeq, conf))  #
        Timer3 = time.time()
        # print('mom')  # 目前的版本在处理真实数据时，计算时间过长。
        #momListEST.append(mom(PcdSeq, estPoseSeq, None, conf))
        # momListGT.append(mom(PcdSeq, gtPoseSeq, None, conf))
        Timer4 = time.time()
        # print("time consume:\n mme: {%f}\n mpv: {%f}\n mom {%f}\n" % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3))
        #'''
        Timer5 = time.time()
        # plane based mom
        # 基于当前窗口的中间帧，中存在的候选正交平面，在当前窗口进行求解
        #PlaneMomList.append(segplane.localOrthPlaneMetric(gtGlobalPcdSeq, colorSeq, orthPlanes))

        Timer6 = time.time()
        # plane normals entropy, PNE
        PNE = segplane.normalClusteringEntropy(candNormals)  #
        normalEntropyList.append(PNE)  # normal entropy for r
        Timer7 = time.time()
        # parallel plane variance, PPV, CoPlane Variance
        #pVariance, pNormals = segplane.parallePlane(candPlanes, candNormals)  # plane variance for t
        PNE = segplane.parallePlane1(candPlanes, candNormals)
        Timer8 = time.time()
        #planeNormalsList.append(pNormals)
        planeVariacneList.append(PNE)

        #print(Timer8-Timer7)
        '''
        with open(outFileName, 'a') as sfile:
            sfile.writelines("time consume:\n mme: %f\n mpv: %f\n mom %f\n SPV: %f\n PNE: %f\n CPV+CPN: %f\n"
                             % (Timer2-Timer1, Timer3-Timer2, Timer4-Timer3, Timer6-Timer5, Timer7-Timer6, Timer8-Timer7))
        '''
        #with open(outFileName, 'a') as sfile:
        #    sfile.writelines("time consume:\n mom %f\n" % (Timer4-Timer3))
    normalEntropyList = normalized(normalEntropyList)
    planeVariacneList = normalized(planeVariacneList)
    rpeTList = normalized(rpeTList)
    rpeRList = normalized(rpeRList)
    # 以时间窗口为单位进行输出
    with open(outFileName, 'a') as sfile:
        #print("window:", win, mmeList, mpvList, momList, rpeFullList, rpeTList, rpeRList)
        sfile.writelines("window: %d\n" % (win))

        sfile.writelines("noise: \n")
        sfile.writelines("GT Noise: "+str(errList)+"\n")

        sfile.writelines("Coordinate: \n")
        sfile.writelines("x-y-z: "+str(coordList).replace('[', '').replace(']', '').replace('), (', ':')+"\n")

        sfile.writelines("map-based metrics:\n")
        sfile.writelines("MME-est: "+str(mmeListEST)+"\n")
        sfile.writelines("MPV-est: "+str(mpvListEST)+"\n")
        sfile.writelines("MOM-est: "+str(momListEST)+"\n")
        #sfile.writelines("MME-gt: "+str(mmeListGT)+"\n")
        #sfile.writelines("MPV-gt: "+str(mpvListGT)+"\n")
        #sfile.writelines("MOM-gt: "+str(momListGT)+"\n")

        sfile.writelines("reference-based metrics:\n")
        sfile.writelines("RPE-Full: "+str(rpeFullList)+"\n")
        sfile.writelines("RPE-T: "+str(rpeTList)+"\n")
        sfile.writelines("RPE-R: "+str(rpeRList)+"\n")

        sfile.writelines("new-plane-based metrics:\n")

        sfile.writelines("Plane-MOM: "+str(PlaneMomList)+"\n")
        sfile.writelines("PNE: "+str(normalEntropyList).replace(',', ' ').replace('[', '').replace(']', '')+"\n")
        sfile.writelines("CPV: "+str(planeVariacneList).replace(',', ' ').replace('[', '').replace(']', '')+"\n")
        sfile.writelines("CPN: "+str(planeNormalsList)+"\n")


#def multi_run_wrapper_windowSampling(args):
#    return windowSampling(*args)
'''
真实数据的noise不需要生成，直接用两个pose进行对比，但是地图至少需要两帧进行计算
'''
if __name__ == '__main__':
    fatherPath = r'J:\map\realgarage'
    scene = '5-slam' # key frame at 5-meters
    #noiseTest()
    #aggregateMap()
    # multi-porcess with diff windowSize
    win = 3
    # 输入到自己的目录下
    #outFileName = r"J:\map\realgarage\results\win=%d.txt" % (win)
    outFileName = r"J:\map\realgarage\results\withcoords.txt"

    #windowSamplingMulitPorcess(fatherPath, scene, win, noisemode, scale, outFileName)
    windowSamplingMulitPorcessCurrent(fatherPath, scene, outFileName)
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


