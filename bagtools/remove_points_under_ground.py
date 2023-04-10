import numpy as np
import open3d as o3d
import os
import pypcd
import glob
import shutil

input_path = '/home/kid/dataset/map_garage'
output_path = '/media/kid/info/map/realgarage/map_garage'

if not os.path.exists(output_path):
    os.makedirs(output_path)


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

# x, y, z
def boundary_filter(pc, color, lim = [30, 30, 0]):
    #
    filtered_pc = pc[np.where(
        (abs(pc[:, 0]) < lim[0]) &  # x
        (abs(pc[:, 0]) < lim[1]) &  # y
        (pc[:, 2] < lim[2])  # z, under the sensor
    )]
    filtered_color = color[np.where(  # index
        (abs(pc[:, 0]) < lim[0]) &
        (abs(pc[:, 0]) < lim[1]) &
        (pc[:, 2] < lim[2])
    )]

    return filtered_pc, filtered_color


def get_ground_param(pcd):
    '''
    :param pcd:
    :return: the center point and the normalized normal vector
    distance(pc, current plane) = (pc-center).dot(normal)>-1
    '''
    while np.asarray(pcd.points).shape[0] > 300:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.03,
                                                 ransac_n=3,
                                                 num_iterations=100)
        [a, b, c, d] = plane_model
        if abs(a) < 0.1 and abs(b) < 0.1 and abs(c) > 0.95:
            inlier_cloud = pcd.select_by_index(inliers)
            # pc_show([inlier_cloud])
            center_point = np.mean(np.asarray(inlier_cloud.points), axis=0)
            return center_point, [a, b, c]
            # break
        else:
            pcd = pcd.select_by_index(inliers, invert=True)

def remove_points_under_plane(point_arr, color_arr, p_center, p_normal, epsilon=-0.05):
    new_center = np.ones_like(point_arr) * p_center
    # new_normal = np.ones_like(point_arr) * p_normal  # index only calculated once
    idx = point_arr - new_center
    idx = np.dot(idx, p_normal) > epsilon  # > 保留上面, < 保留下面
    filtered_pc = point_arr[idx]
    filtered_color = color_arr[idx]
    #'''
    #  open3d visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pc)
    pcd.colors = o3d.utility.Vector3dVector(filtered_color)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_arr)
    # pcd1.colors = o3d.utility.Vector3dVector(color_arr)
    pcd1.paint_uniform_color([1, 0, 0])
    pc_show([pcd1, pcd])
    #'''
    return filtered_pc, filtered_color


def get_frames(path):
    #subfolders = os.listdir(path)
    # print(subfolders)
    pcd_list = glob.glob(path+"/*.pcd")
    #for folder in pcd_list:
    #    print(folder)
    #    if os.path.isdir(os.path.join(path, folder)):
    for file in pcd_list:
        # file_name = os.path.join(path, file)
        pc = pypcd.PointCloud.from_path(file)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32) / 256
        points = np.transpose(np.vstack((np_x, np_y, np_z)))
        colors = np.transpose(np.vstack((np_i, np_i, np_i)))
        points_boundary, colors_boundary = boundary_filter(points, colors)  # filtering sparse cloud for far distance and hight
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_boundary)
        pcd.colors = o3d.utility.Vector3dVector(colors_boundary)
        plane_center, plane_normal = get_ground_param(pcd)
        f_points, f_colors = remove_points_under_plane(points, colors, plane_center, plane_normal)
        # print(f_points.shape, f_colors[:, 0].reshape(-1, 1).shape)
        new_data = np.hstack((f_points, f_colors[:, 0].reshape(-1, 1)))
        # print(new_data.shape)
        f_pc = pypcd.make_xyzi_point_cloud(new_data)
        file_name = file.split('/')[-1]
        full_name = os.path.join(output_path, file_name)
        #f_pc.save_pcd(full_name, compression='binary_compressed')
        #shutil.copy(file.replace('.pcd', '.odom'), full_name.replace('.pcd', '.odom'))
        #pc_show([pcd])


get_frames(input_path)