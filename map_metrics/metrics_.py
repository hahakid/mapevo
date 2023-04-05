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
import copy
import numpy as np
import open3d as o3d

from typing import Optional, Type, Any, List, Callable
from nptyping import NDArray

from map_metrics.utils.orthogonal_ import extract_orthogonal_subsets
from map_metrics.config import BaseConfig, LidarConfig

__all__ = ["aggregate_map", "mme", "mpv", "mom"]


def aggregate_map(pcs: List[o3d.geometry.PointCloud], ts: List[NDArray[(4, 4), np.float64]]) -> o3d.geometry.PointCloud:
    """
    Build a map from point clouds with their poses

    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)

    Returns
    -------
    pc_map: o3d.geometry.PointCloud
        Map aggregated from point clouds

    Raises
    ------
    ValueError
        If number of point clouds does not match number of poses
    """
    if len(pcs) != len(ts):
        raise ValueError("Number of point clouds does not match number of poses")

    ts = [np.linalg.inv(ts[0]) @ T for T in ts]
    pc_map = o3d.geometry.PointCloud()
    for i, pc in enumerate(pcs):
        pc_map += copy.deepcopy(pc).transform(ts[i])

    return pc_map


def _plane_variance(points: NDArray[(Any, 3), np.float64]) -> float:
    """
    Compute plane variance of given points

    Parameters
    ----------
    points: NDArray[(Any, 3), np.float64]
        Point Cloud points

    Returns
    -------
    plane_variance: float
        Points plane variance
    """
    cov = np.cov(points.T)
    eigenvalues = np.linalg.eig(cov)[0]
    return min(eigenvalues)


def _entropy(points: NDArray[(Any, 3), np.float64]) -> Optional[float]:
    """
    Compute entropy of given points

    Parameters
    ----------
    points: NDArray[(Any, 3), np.float64]
        Point Cloud points

    Returns
    -------
    entropy: Optional[float]
        Points entropy
    """
    cov = np.cov(points.T)  # 求行列式，
    det = np.linalg.det(2 * np.pi * np.e * cov)
    if det > 0:  # det<1 返回负值
        return 0.5 * np.log(det)

    return None


def _mean_map_metric(
    pcs: List[o3d.geometry.PointCloud],
    ts: List[NDArray[(4, 4), np.float64]],
    config: Type[BaseConfig] = LidarConfig,
    alg: Callable = _plane_variance,
) -> float:
    """
    No-reference metric algorithms helper

    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    config: BaseConfig
        Scene hyperparameters
    alg: Callable
        Metric algorithm basis (e.g., plane variance, entropy)
    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    pc_map = aggregate_map(pcs, ts)

    map_tree = o3d.geometry.KDTreeFlann(pc_map)
    points = np.asarray(pc_map.points)
    metric = []
    for i in range(points.shape[0]):
        point = points[i]
        _, idx, _ = map_tree.search_radius_vector_3d(point, config.KNN_RAD)
        if len(idx) > config.MIN_KNN:
            metric_value = alg(points[idx])
            if metric_value is not None:
                metric.append(metric_value)

    return 0.0 if len(metric) == 0 else np.mean(metric)

def _mean_map_metric_map(
        pcs: o3d.geometry.PointCloud,
        config: Type[BaseConfig] = LidarConfig,
        alg: Callable = _plane_variance,
) -> float:
    """
    No-reference metric algorithms helper

    Parameters
    ----------
    pcs: o3d.geometry.PointCloud
        a map
    config: BaseConfig
        Scene hyperparameters
    alg: Callable
        Metric algorithm basis (e.g., plane variance, entropy)
    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """

    map_tree = o3d.geometry.KDTreeFlann(pcs)
    points = np.asarray(pcs.points)
    metric = []
    for i in range(points.shape[0]):
        point = points[i]
        _, idx, _ = map_tree.search_radius_vector_3d(point, config.KNN_RAD)
        if len(idx) > config.MIN_KNN:
            metric_value = alg(points[idx])
            if metric_value is not None:
                metric.append(metric_value)

    return 0.0 if len(metric) == 0 else np.mean(metric)


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

def _orth_mpv(
    pcs: List[o3d.geometry.PointCloud],
    ts: List[NDArray[(4, 4), np.float64]],
    config: Type[BaseConfig] = LidarConfig,
    orth_list: List[o3d.geometry.PointCloud] = None,
):
    """

    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    config: BaseConfig
        Scene hyperparameters
    orth_list: List[o3d.geometry.PointCloud], default=None
        List of orthogonal planes of the map

    Returns
    -------
    val: float
        The value of MPV computed on orthogonal planes of the map
    """
    pc_map = aggregate_map(pcs, ts)  # 叠加地图
    # pc_show([pc_map])
    map_tree = o3d.geometry.KDTreeFlann(pc_map)  # 构建kd-tree
    points = np.asarray(pc_map.points)  # 转array

    if orth_list is None:
        # 改为中间一帧
        #
        # pc = pcs[0]  # 正交平面候选集只基于第一帧点云进行了计算？
        pc = pcs[int(len(pcs)/2)]  # 因为我们给了一个序列，从中间开始比较合理
        orth_list, _, _ = extract_orthogonal_subsets(pc, config=config, eps=1e-1)

    ''' only for visualization
    ppc = np.vstack(orth_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ppc[:, :3])
    pc_show([pcd])
    '''

    orth_axes_stats = []

    for chosen_points in orth_list:
        metric = []
        for i in range(np.asarray(chosen_points).shape[0]):
            point = chosen_points[i]
            _, idx, _ = map_tree.search_radius_vector_3d(point, config.KNN_RAD)
            # TODO: add 3 to config
            if len(idx) > 3:
                metric.append(_plane_variance(points[idx]))

        avg_metric = np.median(metric)
        orth_axes_stats.append(avg_metric)

    return np.sum(orth_axes_stats)

def _orth_mpv_map(
        pcs: o3d.geometry.PointCloud,
        dc_pcs: o3d.geometry.PointCloud,
        config: Type[BaseConfig] = LidarConfig,
        orth_list: List[o3d.geometry.PointCloud] = None,
):
    """

    Parameters
    ----------
    pcs: o3d.geometry.PointCloud
        a map
    dc_pcs: o3d.geometry.PointCloud
        a downsampled map
    config: BaseConfig
        Scene hyperparameters
    orth_list: List[o3d.geometry.PointCloud], default=None
        List of orthogonal planes of the map

    Returns
    -------
    val: float
        The value of MPV computed on orthogonal planes of the map
    """
    map_tree = o3d.geometry.KDTreeFlann(pcs)  # 构建kd-tree from pcs
    points = np.asarray(pcs.points)  # 转array

    # search orth plane candidate points in the downsample point cloud
    if orth_list is None:
        orth_list, _, _ = extract_orthogonal_subsets(dc_pcs, config=config, eps=1e-1)

    ''' only for visualization
    ppc = np.vstack(orth_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ppc[:, :3])
    pc_show([pcd])
    '''
    orth_axes_stats = []
    # search the local planes based on orth_list on pcs
    for chosen_points in orth_list:
        metric = []
        for i in range(np.asarray(chosen_points).shape[0]):
            point = chosen_points[i]
            _, idx, _ = map_tree.search_radius_vector_3d(point, config.KNN_RAD)
            # TODO: add 3 to config
            if len(idx) > 3:
                metric.append(_plane_variance(points[idx]))

        avg_metric = np.median(metric)
        orth_axes_stats.append(avg_metric)

    return np.sum(orth_axes_stats)


def mme(
    pcs: List[o3d.geometry.PointCloud],
    ts: List[NDArray[(4, 4), np.float64]],
    config: Type[BaseConfig] = LidarConfig,
) -> float:
    """
    Mean Map Entropy
    A no-reference metric algorithm based on entropy

    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    config: BaseConfig
        Scene hyperparameters

    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _mean_map_metric(pcs, ts, config, alg=_entropy)

def mme_map(
        pcs: o3d.geometry.PointCloud,
        config: Type[BaseConfig] = LidarConfig,
) -> float:
    """
    Mean Map Entropy
    A no-reference metric algorithm based on entropy

    Parameters
    ----------
    pcs: o3d.geometry.PointCloud
        a map
    config: BaseConfig
        Scene hyperparameters

    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _mean_map_metric_map(pcs, config, alg=_entropy)


def mpv(
    pcs: List[o3d.geometry.PointCloud],
    ts: List[NDArray[(4, 4), np.float64]],
    config: Type[BaseConfig] = LidarConfig,
) -> float:
    """
    Mean Plane Variance
    A no-reference metric algorithm based on plane variance

    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    config: BaseConfig
        Scene hyperparameters

    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _mean_map_metric(pcs, ts, config, alg=_plane_variance)

def mpv_map(
        pcs: o3d.geometry.PointCloud,
        config: Type[BaseConfig] = LidarConfig,
) -> float:
    """
    Mean Plane Variance
    A no-reference metric algorithm based on plane variance

    Parameters
    ----------
    pcs: o3d.geometry.PointCloud
        a map
    config: BaseConfig
        Scene hyperparameters

    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _mean_map_metric_map(pcs, config, alg=_plane_variance)



def mom(
    pcs: List[o3d.geometry.PointCloud],
    ts: List[NDArray[(4, 4), np.float64]],
    orth_list: List[o3d.geometry.PointCloud] = None,
    config: Type[BaseConfig] = LidarConfig,
):
    """
    Mutually Orthogonal Metric
    A no-reference metric algorithm based on MPV on orthogonal planes subset

    Parameters
    ----------
    pcs: List[o3d.geometry.PointCloud]
        Point Clouds obtained from sensors
    ts: List[NDArray[(4, 4), np.float64]]
        Transformation matrices list (i.e., Point Cloud poses)
    orth_list: List[o3d.geometry.PointCloud], default=None
        List of orthogonal planes of the map
    config: BaseConfig
        Scene hyperparameters

    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _orth_mpv(pcs, ts, config, orth_list=orth_list)

def mom_map(
        pcs: o3d.geometry.PointCloud,
        dc_pcs: o3d.geometry.PointCloud,
        orth_list: List[o3d.geometry.PointCloud] = None,
        config: Type[BaseConfig] = LidarConfig,
):
    """
    Mutually Orthogonal Metric
    A no-reference metric algorithm based on MPV on orthogonal planes subset

    Parameters
    ----------
    pcs: o3d.geometry.PointCloud
        a map
    dc_pcs: o3d.geometry.PointCloud
        a downsampled map
    orth_list: List[o3d.geometry.PointCloud], default=None
        List of orthogonal planes of the map
    config: BaseConfig
        Scene hyperparameters

    Returns
    -------
    mean: float
        Mean of given metric algorithm values
    """
    return _orth_mpv_map(pcs, dc_pcs, config, orth_list=orth_list)