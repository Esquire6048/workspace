# point_cloud_slice.py
"""
点云切分：模拟多摄像机观测 + 可选遮挡 (occlusion) 效果

假设 XZ 平面为水平面，Y 轴为高度。

主要接口
----------
`slice_point_cloud(points, num_sensors, fov_deg, center, sensor_distance, angle_resolution, occlusion, return_masks)`

参数
-----
- points: ndarray (N_pts, 3)
- num_sensors: int
- fov_deg: float, 水平视场角 (°)
- center: 'auto'|'origin'|ndarray
- sensor_distance: float|None
- angle_resolution: int, 水平分辨率 (bin 数)
- occlusion: bool, 是否启用遮挡测试
- return_masks: bool, 是否返回初始 FOV 掩码

返回
----
- slices: list of ndarray
- masks: 可选 ndarray (num_sensors, N_pts)
"""
import numpy as np
from typing import List, Tuple, Union, Optional

__all__ = ['slice_point_cloud']


def slice_point_cloud(
    points: np.ndarray,
    *,
    num_sensors: int = 3,
    fov_deg: float = 60.0,
    center: Union[str, np.ndarray] = 'auto',
    sensor_distance: Optional[float] = None,
    angle_resolution: int = 360,
    occlusion: bool = True,
    return_masks: bool = False,
) -> List[np.ndarray] | Tuple[List[np.ndarray], np.ndarray]:
    """
    模拟多摄像机切分 + 屏蔽测试。
    """
    # 校验
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be (N_pts,3)')
    if not (0.0 < fov_deg <= 180.0):
        raise ValueError('fov_deg must in (0,180]')
    if angle_resolution < 1:
        raise ValueError('angle_resolution must be >=1')

    # 中心
    if isinstance(center, str):
        cstr = center.lower()
        if cstr == 'auto':
            c = points.mean(axis=0)
        elif cstr == 'origin':
            c = np.zeros(3, dtype=float)
        else:
            raise ValueError('center must be auto/origin/ndarray')
    else:
        c = np.asarray(center, dtype=float)
        if c.shape != (3,):
            raise ValueError('center ndarray must shape (3,)')

    # 计算水平半径 (XZ)
    horiz = points[:, [0,2]] - c[[0,2]]
    dists_h = np.linalg.norm(horiz, axis=1)
    R = float(sensor_distance) if sensor_distance is not None else float(dists_h.max())

    # 摄像机位置 + 方向
    az = np.linspace(0.0, 2*np.pi, num_sensors, endpoint=False)
    sx = c[0] + R * np.cos(az)
    sz = c[2] + R * np.sin(az)
    sy = np.full(num_sensors, c[1])
    sensor_pos = np.stack([sx, sy, sz], axis=1)
    # 方向向量指向中心
    dirs = c - sensor_pos
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # 初始 FOV 掩码
    # rel_dir: (S, N, 3)
    rel_dir = points[None, :, :] - sensor_pos[:, None, :]
    rel_norm = np.linalg.norm(rel_dir, axis=2, keepdims=True)
    rel_norm[rel_norm==0] = 1e-8
    rel_unit = rel_dir / rel_norm
    cos_th = np.cos(np.deg2rad(fov_deg/2.0))
    masks = np.einsum('sd,spd->sp', dirs, rel_unit) >= cos_th  # (S,N)

    slices: List[np.ndarray] = []
    # 上方向用于计算水平角
    up = np.array([0.0, 1.0, 0.0], dtype=float)

    for i in range(num_sensors):
        mask_i = masks[i]
        idx = np.nonzero(mask_i)[0]
        if idx.size == 0:
            slices.append(np.empty((0,3)))
            continue
        pts_i = points[idx]
        if not occlusion:
            slices.append(pts_i)
            continue
        # occlusion: 深度测试
        rel_i = pts_i - sensor_pos[i]
        dists = np.linalg.norm(rel_i, axis=1)
        # 水平右向量
        right = np.cross(dirs[i], up)
        # 计算水平角 θ = atan2(rel·right, rel·dirs)
        proj_right = rel_i.dot(right)
        proj_forward = rel_i.dot(dirs[i])
        theta = np.degrees(np.arctan2(proj_right, proj_forward))  # (-180,180]
        # 分桶
        half = fov_deg/2.0
        # normalize theta to [0,fov]
        theta_clamped = np.clip(theta, -half, half) + half
        bin_idx = (theta_clamped / fov_deg * angle_resolution).astype(int)
        bin_idx = np.clip(bin_idx, 0, angle_resolution-1)
        # 对每个桶只保留最近的点
        keep_idx = []
        for b in range(angle_resolution):
            inds = np.where(bin_idx==b)[0]
            if inds.size==0: continue
            local = inds[np.argmin(dists[inds])]
            keep_idx.append(local)
        if keep_idx:
            slices.append(pts_i[keep_idx])
        else:
            slices.append(np.empty((0,3)))

    return (slices, masks) if return_masks else slices
