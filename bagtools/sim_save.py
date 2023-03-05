# !/usr/bin/env python2
# coding=utf-8
from __future__ import absolute_import, print_function

import os

import ros_numpy
import rospy
import message_filters
import tf

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, Image, PointField, Imu
import numpy as np
import numpy.linalg as LA
import transforms3d

# 保存路径需要修改一下，后面就是用point labeler打开这个目录
root = "/media/kid/info/map/savedframes/floor2"

if not os.path.exists(root):
    os.mkdir(root)

is_first_frame = True
timestamps = []
poses = []
frame_id = 0
first_time = 0.0
first_RT = np.zeros([4, 4], dtype=np.float32)
first_traj = np.zeros([3, 1], dtype=np.float32)


# R_axes = np.eye(4)
# R_axes[:3, :3] = transforms3d.euler.euler2mat(np.pi, 0, 0, 'sxyz')


def get_RT_matrix(odom_msg):
    quaternion = np.asarray(
        [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
         odom_msg.pose.pose.orientation.z])
    translation = np.asarray(
        [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])

    rotation = transforms3d.quaternions.quat2mat(quaternion)

    # print(T_qua2rota)
    RT_matrix = np.eye(4)
    RT_matrix[:3, :3] = rotation
    RT_matrix[:3, 3] = translation.T

    # RT_matrix = np.matmul(R_axes, RT_matrix)

    return RT_matrix


def cb_sync_position(pc_msg, odom_msg):
    global is_first_frame, frame_id, first_time, first_RT, first_traj
    if is_first_frame:
        first_time = pc_msg.header.stamp.to_sec()
        first_RT = get_RT_matrix(odom_msg)
        is_first_frame = False

        first_traj = first_RT[:3, 3].reshape((3, 1))

    sec = pc_msg.header.stamp.secs
    nsec = pc_msg.header.stamp.nsecs

    present_RT = get_RT_matrix(odom_msg)
    present_traj = present_RT[:3, 3].reshape((3, 1))

    RT_matrix = np.matmul(present_RT, np.linalg.inv(first_RT))
    # RT_matrix = np.matmul(np.linalg.inv(first_RT), present_RT)

    # RT_matrix = present_RT

    kitti_RT = RT_matrix[:3, :].reshape(-1)

    delta_traj = present_traj - first_traj
    RT_matrix[:3, :] = delta_traj
    RT_matrix[:3, :3] = np.eye(3)

    odom = Odometry()
    quaternion = transforms3d.quaternions.mat2quat(RT_matrix[:3, :3])
    position = RT_matrix[:3, 3].T
    odom.header.stamp = pc_msg.header.stamp
    odom.header.frame_id = 'map'
    odom.child_frame_id = 'velodyne'
    odom.pose.pose.orientation.w = quaternion[0]
    odom.pose.pose.orientation.x = quaternion[1]
    odom.pose.pose.orientation.y = quaternion[2]
    odom.pose.pose.orientation.z = quaternion[3]
    odom.pose.pose.position.x = position[0]
    odom.pose.pose.position.y = position[1]
    odom.pose.pose.position.z = position[2]
    odom_publisher.publish(odom)

    br = tf.TransformBroadcaster()
    t = TransformStamped()
    t.header.frame_id = odom.header.frame_id
    t.header.stamp = odom.header.stamp
    t.child_frame_id = odom.child_frame_id
    t.transform.translation.x = odom.pose.pose.position.x
    t.transform.translation.y = odom.pose.pose.position.y
    t.transform.translation.z = odom.pose.pose.position.z

    t.transform.rotation.x = odom.pose.pose.orientation.x
    t.transform.rotation.y = odom.pose.pose.orientation.y
    t.transform.rotation.z = odom.pose.pose.orientation.z
    t.transform.rotation.w = odom.pose.pose.orientation.w
    br.sendTransformMessage(t)

    # return

    pc_array = ros_numpy.numpify(pc_msg)
    if len(pc_array.shape) == 2:
        pc = np.zeros((pc_array.shape[0] * pc_array.shape[1], 4))
    else:
        pc = np.zeros((pc_array.shape[0], 4))
    # 解析点格式
    pc[:, 0] = pc_array['x'].reshape(-1)
    pc[:, 1] = pc_array['y'].reshape(-1)
    pc[:, 2] = pc_array['z'].reshape(-1)
    pc[:, 3] = pc_array['intensity'].reshape(-1)
    pc = pc[~np.isnan(pc).any(axis=1)]  # remove column with nan
    # pc[:, 4] = pc_array['ring'].reshape(-1)
    # pc[:, 5] = pc_array['time'].reshape(-1)

    # pc = pc[~np.isnan(pc).any(axis=1), :]
    # pc = np.nan_to_num(pc)
    #
    # pc = pc[np.where(
    #     # (pc[:, 0] > 0.2) | (pc[:, 0] < -2.0) &
    #     # (pc[:, 1] > 1) | (pc[:, 1] < -1.0) &
    #     # (pc[:, 2] > -1.4)
    # )]

    print("Saving No.", frame_id, "frame, and the size of pointcloud in this frame is: ", pc.shape[0], ", time is: ",
          str(int(sec)) + str(format(float(nsec) / 1e9, '.9f'))[1:])

    pc_file = pc[:, :4].reshape(-1).astype(np.float32)
    save = os.path.join(root, 'velodyne')
    if not os.path.exists(save):
        os.mkdir(save)
    save_file = os.path.join(save, '{:06d}.bin'.format(frame_id))
    pc_file.tofile(save_file)

    # raw_pc = pc.reshape(-1).astype(np.float32)
    # save = os.path.join(root, 'velo_raw')
    # if not os.path.exists(save):
    #     os.mkdir(save)
    # raw_file = os.path.join(save, '{:06d}.bin'.format(frame_id))
    # raw_pc.tofile(raw_file)

    # print("times frame number :", len(timestamps), "--poses frame number: ", len(poses))
    with open(os.path.join(root, "times.txt"), "a") as f1:
        f1.write(str(int(sec)) + str(float(nsec) / 1e9)[1:] + "\n")
    with open(os.path.join(root, "poses.txt"), "a") as f2:
        for i in range(kitti_RT.shape[0] - 1):
            f2.write(str(kitti_RT[i]) + " ")
        f2.write(str(kitti_RT[-1]) + "\n")

    f1.close()
    f2.close()

    frame_id += 1


# def imu_handle(imu_msg):
#


if __name__ == '__main__':
    rospy.init_node('save_data')

    # 订阅
    # points_subscriber = message_filters.Subscriber('/lidar_ins_deskew/cloud_deskewed/velo', PointCloud2)
    # points_subscriber = message_filters.Subscriber('/lio_sam/deskew/fullcloud', PointCloud2)
    points_subscriber = message_filters.Subscriber('/velodyne_points', PointCloud2)
    # raw_point_subscriber = message_filters.Subscriber('/velodyne_points', PointCloud2)
    # points_subscriber = message_filters.Subscriber("/cloud_out", PointCloud2)
    # odom_subscriber = message_filters.Subscriber('/lidar_ins_deskew/sync_odom', Odometry)
    odom_subscriber = message_filters.Subscriber('/state_estimation', Odometry)
    # odom_subscriber = message_filters.Subscriber('/Odometry', Odometry)
    # img_subscriber = message_filters.Subscriber('/image_color', Image)
    # rospy.Subscriber('/imu/data', Imu, callback=imu_handle)

    # points_publisher = rospy.Publisher('/sync/points', PointCloud2, queue_size=1)
    odom_publisher = rospy.Publisher('/test', Odometry, queue_size=5)
    # 时间同步器 完全同步
    #ts = message_filters.TimeSynchronizer(
    #    [points_subscriber, odom_subscriber],
    #    queue_size=5000)
    # 兼容微弱差异
    ts = message_filters.ApproximateTimeSynchronizer(
        [points_subscriber, odom_subscriber], queue_size=20, slop=0.05,
        allow_headerless=True)

    ts.registerCallback(cb_sync_position)

    rospy.loginfo('INIT...')
    rospy.spin()

    # np.savetxt(os.path.join(root, "times.txt"), np.asarray(timestamps).astype(np.float))
    # np.savetxt(os.path.join(root, "poses.txt"), np.asarray(poses).astype(np.float))

    print("Done\n")
