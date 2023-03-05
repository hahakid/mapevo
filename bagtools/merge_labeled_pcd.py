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

semantic_mapping = {  # bgr
    0: [0, 0, 0],  # "unlabeled", and others ignored
    1: [0, 0, 255],  # outliner
    10: [245, 150, 100],  # "floor"
    11: [245, 230, 100],  # "wall"
    12: [150, 60, 30],  # "pillar"
    13: [30, 30, 255],  # "celling"
    21: [180, 30, 80],  # "desk"
    22: [255, 0, 0],  # "chair"
    #7: [200, 40, 255],  # "bicyclist"
    #8: [90, 30, 150],  # "motorcyclist"
    #9: [255, 0, 255],  # "road"
    #10: [255, 150, 255],  # "parking"
    #11: [75, 0, 75],  # "sidewalk"
    #12: [75, 0, 175],  # "other-ground"
    #13: [0, 200, 255],  # "building"
    #14: [50, 120, 255],  # "fence"
    #15: [0, 175, 0],  # "vegetation"
    #16: [0, 60, 135],  # "trunk"
    #17: [80, 240, 150],  # "terrain"
    #18: [150, 240, 255],  # "pole"
    #19: [0, 0, 255]  # "traffic-sign"
}

# root = "/media/hyx/701e2c63-271c-466f-a047-f683746987da/Segmantic_kitti/dataset/sequences"
# 修改数据的存储路径
root = "/media/kid/info/map/savedframes/"
sequence = "garage"


def load_calib(calib_file_path):
    """
    load calibration file(KITTI object format)
    :param calib_file_path:
    :return:
    """
    calib_file = open(calib_file_path, 'r').readlines()
    calib_file = [line
                      .replace('Tr_velo_to_cam', 'Tr_velo_cam')
                      .replace('Tr:', 'Tr_velo_cam:')
                      .replace('R0_rect', 'R_rect')
                      .replace('\n', '')
                      .replace(':', '')
                      .split(' ')
                  for line in calib_file]
    calib_file = {line[0]: [float(item) for item in line[1:] if item != ''] for line in calib_file if len(line) > 1}
    return calib_file


def parse_calib_file(calib_file):
    """
    parse calibration file to calibration matrix
    :param calib_file:
    :return:
    """

    # 外参矩阵
    Tr_velo_cam = np.array(calib_file['Tr_velo_cam']).reshape(3, 4)
    Tr_velo_cam = np.concatenate([Tr_velo_cam, [[0, 0, 0, 1]]], axis=0)
    # 矫正矩阵
    if 'R_rect' in calib_file:
        R_rect = np.array(calib_file['R_rect']).reshape(3, 3)
        R_rect = np.pad(R_rect, [[0, 1], [0, 1]], mode='constant')
        R_rect[-1, -1] = 1
    else:
        R_rect = np.eye(4)
    # 内参矩阵
    P2 = np.array(calib_file['P2']).reshape(3, 4)

    return np.matmul(np.matmul(P2, R_rect), Tr_velo_cam)


def load_kitti_odometry_poses(file):
    """
    :param file: The odometry file path
    :return: R, T matrix
    """
    poses_raw = np.loadtxt(file, dtype=np.float)
    poses = [np.row_stack([p.reshape(3, 4), [0, 0, 0, 1]]) for p in poses_raw]
    return poses


def load_pc(bin_file_path):
    """
    load pointcloud file (velodyne format)
    :param bin_file_path:
    :return:
    """
    # with open(bin_file_path, 'rb') as bin_file:
    #     pc = array.array('f')
    #     pc.frombytes(bin_file.read())
    #     pc = np.array(pc).reshape(-1, 4)
    #     return pc

    pc = np.fromfile(bin_file_path, dtype=np.float32).reshape((-1, 4))
    return pc

def load_label(label_file_path):
    label = np.fromfile(label_file, dtype=np.uint32)
    label = label.reshape((-1))
    return label

def rigid_translate(pc_input, extrinsic):
    # projection
    pc = np.row_stack([pc_input[:, :3].T, np.ones_like(pc_input[:, 0])])
    pc = np.matmul(extrinsic, pc)
    pc = np.row_stack([pc[:3, :], pc_input[:, 3]]).T
    return pc

def get_pcd_from_numpy(pcd_np, color_data):
    pcd = o3d.geometry.PointCloud()
    # color = plt.get_cmap('hot')(color_data / color_data.max())[:, :3]
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(color_data[:, :3])
    #o3d.utility.Vector3dVector(pcd_np[:, :3])
    return pcd

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

def vis(pc, color_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('hot')(color_data / color_data.max())[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('gist_ncar_r')(color_data / color_data.max())[:, :3])
    # pcd = pcd.voxel_down_sample(voxel_size=0.6)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2
    vis.run()
    vis.destroy_window()

def getcolor(lab):
    color = []
    for i in lab:
        #print(i)
        #print(semantic_mapping[i])
        color.append(semantic_mapping[i])
    color = np.asarray(color) / 255.0
    return color
# # load calibration
# calib = load_calib(os.path.join(root, sequence, "calib.txt"))
#
# # # lidar to camera extrinsic
# extrinsic = np.row_stack([np.array(calib['Tr_velo_cam']).reshape(3, 4), [0, 0, 0, 1]])
# extrinsic = np.eye(4)
# extrinsic[:3, :3] = transforms3d.euler.euler2mat(np.pi, 0, 0)
# extrinsic[:3, 3] = np.array([-0.4, 0, -0.4])

# extrinsic = np.array([1, 0, 0, -0.75,
#                       0, 1, 0, -0.09,
#                       0, 0, 1, -1.0342136564577902e+00,
#                       0., 0., 0., 1.]).reshape(4, 4)
# extrinsic[:3, :3] = transforms3d.euler.euler2mat(np.pi-0.05, 0, 0.0495)

# extrinsic = np.array([1, 0, 0, -0.75,
#                       0, 1, 0, -0.00,
#                       0, 0, 1, -1.0342136564577902e+00,
#                       0., 0., 0., 1.]).reshape(4, 4)
# extrinsic[:3, :3] = transforms3d.euler.euler2mat(np.pi-0.05, 0.00, 0.0455)


# extrinsic = np.array([9.9833852937322298e-01, 5.5871057648239582e-02,
#                       -1.4092752966591933e-02, -8.1711404293961420e-01,
#                       5.5719774828092240e-02, -9.9838685405774941e-01,
#                       -1.0908544255671494e-02, -8.3583660816657183e-03,
#                       -1.4679491204295724e-02, 1.0105175007803367e-02,
#                       -9.9984118637713948e-01, -1.0642136564577902e+00, 0., 0., 0., 1.]).reshape(4, 4)
extrinsic = np.eye(4)
# print(extrinsic)

# load poses
poses = load_kitti_odometry_poses(file=os.path.join(root, sequence, 'poses.txt'))
mapping = []
color_all = []
l = len(os.listdir(os.path.join(root, sequence, 'velodyne')))

for idx in range(0, l, 1):
    print(idx)
    # load pc
    pc_file = os.path.join(root, sequence, 'velodyne', '{:06d}.bin'.format(idx))
    label_file = os.path.join(root, sequence, 'labels', '{:06d}.label'.format(idx))
    pc = load_pc(pc_file)
    label = load_label(label_file)
    colors = getcolor(label)

    # pc = pc[~np.isnan(pc).any(axis=1)]  # remove column with nan
    # pc = pc[np.where(
    #     (abs(pc[:, 0]) < 20) &
    #     (abs(pc[:, 1]) < 20) &
    #     (pc[:, 2] > -1.2)
    #     # (pc[:, 3] > 150.0)
    # )]

    # 1. translate points in camera frame
    pc = rigid_translate(pc, extrinsic)

    # convert to world frame
    # pc = rigid_translate(pc, LA.inv(poses[idx]))
    pc = rigid_translate(pc, poses[idx])

    mapping.append(pc)
    color_all.append(colors)

pc = np.row_stack(mapping)
color_all = np.row_stack(color_all)
#vis(pc, (pc[:, 3] + 10) ** 1.1 + 20)
pcd = get_pcd_from_numpy(pc, color_all)  # (pc[:, 2]) ** 1.1 + 20)
o3d.io.write_point_cloud("./test.pcd", pcd)  # cc打不开可能是 文件权限问题
pc_show([pcd])