# point_cloud_slice.py
"""
核心函数 `slice_point_cloud()`
==============================

更新：将虚拟摄影机放置于与参考中心相同高度的圆周上，并采用**锥形视场 (conical FOV)** 进行切分。

用法示例
--------
```python
from point_cloud_slice import slice_point_cloud
# sensor_distance 可选：默认按点云水平最大半径自动计算
sub_clouds = slice_point_cloud(points,
                                num_sensors=4,
                                fov_deg=45,
                                center='auto',
                                sensor_distance=None)
```

返回值为长度 = `num_sensors` 的子点云列表；设置 `return_masks=True` 可同时返回布尔掩码矩阵。
"""
from typing import List, Tuple, Union, Optional
import numpy as np

__all__ = ["slice_point_cloud"]

# -----------------------------------------------------------------------------
#  公共接口
# -----------------------------------------------------------------------------
def slice_point_cloud(
    points: np.ndarray,
    *,
    num_sensors: int = 3,
    fov_deg: float = 60.0,
    center: Union[str, np.ndarray] = "auto",
    sensor_distance: Optional[float] = None,
    return_masks: bool = False,
) -> List[np.ndarray] | Tuple[List[np.ndarray], np.ndarray]:
    """按锥形视场 (conical FOV) 切分点云。

    Parameters
    ----------
    points : ndarray, shape (N_pts, 3)
        输入点云。
    num_sensors : int, default=3
        传感器数量。
    fov_deg : float, default=60.0
        每个传感器的视场角 (0 < fov ≤ 180)（角度）。
    center : {'auto', 'origin', ndarray}, default='auto'
        角度判断参考中心：
        - 'auto'   → 点云几何中心。
        - 'origin' → (0,0,0)。
        - ndarray  → 用户指定 (3,) 坐标。
    sensor_distance : float or None, default=None
        传感器到中心的水平距离。若为 None，自动设为点云在 XY 平面
        上到中心的最大距离。
    return_masks : bool, default=False
        是否额外返回掩码矩阵 (num_sensors, N_pts)。

    Returns
    -------
    slices : list of ndarray
        切分后的子点云列表，长度等于 num_sensors。
    masks : ndarray, optional
        布尔掩码，shape=(num_sensors, N_pts)，仅在 return_masks=True 时返回。
    """
    # --- 校验 ---------------------------------------------------------------
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是形状 (N_pts, 3) 的 ndarray")
    if not (0.0 < fov_deg <= 180.0):
        raise ValueError("fov_deg 必须位于 (0, 180] 区间")

    # --- 参考中心 -----------------------------------------------------------
    if isinstance(center, str):
        center = center.lower()
        if center == 'auto':
            c = points.mean(axis=0)
        elif center == 'origin':
            c = np.zeros(3, dtype=float)
        else:
            raise ValueError("center 只能是 'auto'、'origin' 或 ndarray")
    else:
        c = np.asarray(center, dtype=float)
        if c.shape != (3,):
            raise ValueError("center ndarray 必须形状为 (3,)")

    # --- 计算 sensor_distance -----------------------------------------------
    horiz = points[:, :2] - c[:2]
    dists = np.linalg.norm(horiz, axis=1)
    R = float(sensor_distance) if sensor_distance is not None else float(dists.max())

    # --- 构造传感器位置和方向 -----------------------------------------------
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_sensors, endpoint=False)
    sx = c[0] + R * np.cos(azimuths)
    sy = c[1] + R * np.sin(azimuths)
    sz = np.full(num_sensors, c[2])
    sensor_pos = np.stack([sx, sy, sz], axis=1)  # (num_sensors, 3)

    dirs = c - sensor_pos                          # (num_sensors, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)  # 单位化

    # --- 计算点相对于每个传感器的单位向量 -------------------------------------
    rel = points[None, :, :] - sensor_pos[:, None, :]       # (N_s, N_pts, 3)
    rel_norm = np.linalg.norm(rel, axis=2, keepdims=True)
    rel_norm[rel_norm == 0] = 1e-8
    rel_unit = rel / rel_norm                               # (N_s, N_pts, 3)

    # --- 锥形 FOV 判定 -------------------------------------------------------
    cos_thresh = np.cos(np.deg2rad(fov_deg / 2.0))
    # 点积：dirs(s,3) × rel_unit(s, N_pts,3) → (s, N_pts)
    dots = np.einsum('sd,spd->sp', dirs, rel_unit)        # (num_sensors, N_pts)
    masks = dots >= cos_thresh                             # 布尔矩阵

    # --- 切分输出 -----------------------------------------------------------
    slices = [points[masks[i]] for i in range(num_sensors)]
    return (slices, masks) if return_masks else slices
