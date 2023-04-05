# Copyright (c) 2022, Skolkovo Institute of Science and Technology (Skoltech)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import networkx as nx
import open3d as o3d

from typing import Type, List
from nptyping import NDArray

from ..config import BaseConfig, LidarConfig
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import matplotlib.pyplot as plt

__all__ = ["extract_orthogonal_subsets", "read_orthogonal_subset"]


def _build_normals_and_lambdas(pc, knn_rad):
    pc_tree = o3d.geometry.KDTreeFlann(pc)  # 不知道为什么又构建一次kd-tree，大概之前的没法传递进来
    points = np.asarray(pc.points)
    main_normals = np.asarray(pc.normals)
    normals = []
    lambdas = []
    new_points = []
    # TODO: Add mutable tqdm bar
    #  100倍系数需要考虑 和 3个点就构成簇 需要斟酌一下，因为计算复杂度太高了
    for i in range(points.shape[0]):  # 按点遍历
        point = points[i]
        _, idx, _ = pc_tree.search_radius_vector_3d(point, knn_rad)
        if len(idx) > 3:
            cov = np.cov(points[idx].T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            idx = eigenvalues.argsort()  # 返回特征值的排序索引下标
            eigenvalues = eigenvalues[idx]  # 使用该索引进行排序，可能有复数形式，所以不能sort？
            # 当一簇点构成平面，最大的两个特征值差不多，且远大于最小的特征值（该值也是平面误差）
            if 100 * eigenvalues[0] < eigenvalues[1]:  # 排序后 最小的特征值，约等于平面误差，
                normals.append(main_normals[i])  # 保存该点的法向
                lambdas.append(eigenvalues[0])  # lambdas存放的平面误差
                new_points.append(point)  # 保留该点
    # 返回满足条件，也就是在平面上的点的法向array, 平面误差 和 点集
    return np.vstack(normals), lambdas, np.vstack(new_points)


def _estimate_normals(pc, knn_rad, max_nn):
    if not pc.has_normals():
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=knn_rad, max_nn=max_nn
            )
        )
    # lambdas并未被使用
    normals, lambdas, new_points = _build_normals_and_lambdas(pc, knn_rad)

    cut_pcd = o3d.geometry.PointCloud()
    cut_pcd.points = o3d.utility.Vector3dVector(new_points)
    cut_pcd.normals = o3d.utility.Vector3dVector(normals)

    return cut_pcd


def _filter_clusters(clustering, normals, min_clust_size):
    n_clusters = np.unique(clustering.labels_).shape[0]
    labels = clustering.labels_
    huge_clusters = []  # 也没有用上
    cluster_means, cluster_means_ind = [], []

    for i in range(n_clusters):
        ind = np.where(labels == i)
        if ind[0].shape[0] > min_clust_size:
            huge_clusters.append(i)
            cluster_means.append(np.mean(np.vstack(normals)[ind], axis=0))
            cluster_means_ind.append(i)

    # Normalize means of every cluster
    cluster_means = np.vstack(cluster_means)
    cluster_means = cluster_means / np.linalg.norm(cluster_means, axis=1)[:, None]

    return cluster_means, cluster_means_ind


def _find_max_clique(labels, cluster_means, cluster_means_ind, eps=1e-1):
    N = cluster_means.shape[0]
    adj_matrix = np.zeros((N, N))
    #  基于法向垂直，构建邻接矩阵
    #  将当前簇与剩余簇将进行遍历，相当于只遍历对角阵，但是最后2句都填充了，是个对称矩阵，主对角线为0
    #   基于共线和正交关系，增加关联
    for i in range(N):  #
        for j in range(i):  #
            x = np.abs(np.dot(cluster_means[i], cluster_means[j]))  # |n_i dot n_j|
            if x < eps:  # nearly orthogonal， 正交平面 原作者这里写反了
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
            if x > 1-eps:  # nearly co-line， 可能是同一平面， 可视化观测时，会发现同一平面上的点的法向正好相反
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
                amount += np.sum(labels == cluster_means_ind[j])  # 累加：对应 cluster 编号的点
            full_cliques_size.append(amount)  # 团的数量
            full_cliques.append(clique)  #

    if len(full_cliques) == 0:
        raise ValueError("Length of full_cliques == 0")
    # 找到其中最大的一个团，输出
    max_ind = full_cliques_size.index(max(full_cliques_size))
    return full_cliques[max_ind]

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

# 基于正交平面集合 和 正交平面法向可视化
def pcdColoringWithLabel2(orthset, orthnorm):
    assert len(orthset) == len(orthnorm)
    pclist = []
    for i in range(len(orthset)):
        cur_points = orthset[i]
        cur_normals = orthnorm[i]
        cur_pcd = o3d.geometry.PointCloud()
        cur_pcd.points = o3d.utility.Vector3dVector(cur_points)
        cur_pcd.normals = o3d.utility.Vector3dVector(cur_normals)
        l = len(cur_points)
        col = np.asarray(plt.get_cmap('gist_ncar_r')(i / len(orthset))).reshape(1, 4)
        #col = np.asarray(plt.get_cmap('hot')(i / n_clusters)).reshape(1, 4)
        cur_pcd.colors = o3d.utility.Vector3dVector(np.full((l, 4), col)[:, :3])
        pclist.append(cur_pcd)
    pc_show(pclist)
    #pc_show(pclist, norm_flag=True)

#idx = 经过过滤后剩余的，簇>5的剩余簇
def pcdColoringWithLabel1(pcd, clustering, idx):
    pc_points = np.asarray(pcd.points)
    pc_normals = np.asarray(pcd.normals)

    n_clusters = np.unique(clustering.labels_).shape[0]
    labels = clustering.labels_
    pclist = []
    for i in idx:
        ind = np.where(labels == i)
        cur_points = pc_points[ind]
        cur_normals = pc_normals[ind]
        cur_pcd = o3d.geometry.PointCloud()
        cur_pcd.points = o3d.utility.Vector3dVector(cur_points)
        cur_pcd.normals = o3d.utility.Vector3dVector(cur_normals)
        l = len(cur_points)
        col = np.asarray(plt.get_cmap('gist_ncar_r')(i / n_clusters)).reshape(1, 4)
        #col = np.asarray(plt.get_cmap('hot')(i / n_clusters)).reshape(1, 4)
        cur_pcd.colors = o3d.utility.Vector3dVector(np.full((l, 4), col)[:, :3])
        pclist.append(cur_pcd)
    pc_show(pclist)
    #pc_show(pclist, norm_flag=True)


def pcdColoringWithLabel(pcd, clustering):
    pc_points = np.asarray(pcd.points)
    pc_normals = np.asarray(pcd.normals)

    n_clusters = np.unique(clustering.labels_).shape[0]
    labels = clustering.labels_
    pclist = []
    for i in range(n_clusters):
        ind = np.where(labels == i)
        cur_points = pc_points[ind]
        cur_normals = pc_normals[ind]
        cur_pcd = o3d.geometry.PointCloud()
        cur_pcd.points = o3d.utility.Vector3dVector(cur_points)
        cur_pcd.normals = o3d.utility.Vector3dVector(cur_normals)
        l = len(cur_points)
        col = np.asarray(plt.get_cmap('gist_ncar_r')(i / n_clusters)).reshape(1, 4)
        #col = np.asarray(plt.get_cmap('hot')(i / n_clusters)).reshape(1, 4)
        cur_pcd.colors = o3d.utility.Vector3dVector(np.full((l, 4), col)[:, :3])
        pclist.append(cur_pcd)
    pc_show(pclist)
    #pc_show(pclist, norm_flag=True)


def extract_orthogonal_subsets(pc, config: Type[BaseConfig] = LidarConfig, eps=1e-1):
    cut_pc = _estimate_normals(pc, knn_rad=config.KNN_RAD, max_nn=config.MAX_NN)
    # pc_show([cut_pc], norm_flag=True)
    normals = np.asarray(cut_pc.normals)
    # 大于0.1m的点不进行合并，凝聚聚类
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-1, compute_full_tree=True
    ).fit(normals)
    labels = clustering.labels_
    # 可视化
    # pcdColoringWithLabel(cut_pc, clustering)

    # 基于簇规模进行过滤，并计算保留簇的归一化的法向均值，以及簇编号
    cluster_means, cluster_means_ind = _filter_clusters(
        clustering, normals, min_clust_size=config.MIN_CLUST_SIZE
    )

    # pcdColoringWithLabel1(cut_pc, clustering, cluster_means_ind)

    # 最大团搜索算法 @ 聚类编号 @ 保留的每个类的法向均值 @ 保留的类的编号 @ 极小阈值
    # 包含所有 正交平面点云
    max_clique = _find_max_clique(labels, cluster_means, cluster_means_ind, eps=eps)

    # Obtain orth subset and normals for those cliques
    pc_points = np.asarray(cut_pc.points)
    orth_subset = [
        pc_points[np.where(labels == cluster_means_ind[i])[0]] for i in max_clique
    ]
    pc_normals = np.asarray(cut_pc.normals)
    orth_normals = [
        pc_normals[np.where(labels == cluster_means_ind[i])[0]] for i in max_clique
    ]
    clique_normals = [cluster_means[i] for i in max_clique]

    #pcdColoringWithLabel2(orth_subset, orth_normals)

    # 计算只是用了 正交集合
    return orth_subset, orth_normals, clique_normals


def read_orthogonal_subset(
    orth_subset_name: Path, orth_pose_name: Path, ts: List[NDArray[(4, 4), np.float64]]
):
    """Read and aggregate an orthogonal subset

    Parameters
    ----------
    orth_subset_name: Path
        Orthogonal subset data
    orth_pose_name: Path
        Pose of orthogonal subset in the map
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)

    Returns
    -------
    orth_subset
        Aggregated orthogonal subset
    """
    orth_list = np.load(str(orth_subset_name), allow_pickle=True)
    orth_pose = np.loadtxt(str(orth_pose_name), usecols=range(4))
    orth_pose = np.linalg.inv(ts[0]) @ orth_pose

    return [
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surface))
        .transform(orth_pose)
        .points
        for surface in orth_list
    ]
