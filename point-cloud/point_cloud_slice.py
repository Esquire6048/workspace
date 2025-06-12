import numpy as np
from typing import List, Tuple, Union


def _sensor_directions(num_sensors: int) -> np.ndarray:
    """
    均匀分布在 XY 平面的单位方向向量，形状 (num_sensors, 3)
    """
    az = np.linspace(0.0, 2.0 * np.pi, num_sensors, endpoint=False)
    return np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)


def slice_point_cloud(
    points: np.ndarray,
    *,
    num_sensors: int = 3,
    fov_deg: float = 60.0,
    center: Union[str, np.ndarray] = "auto",
    return_masks: bool = False,
) -> List[np.ndarray] | Tuple[List[np.ndarray], np.ndarray]:
    """按视场切分点云。

    Parameters
    ----------
    points : ndarray, shape (N_pts, 3)
        输入点云。
    num_sensors : int, default 3
        传感器数量。
    fov_deg : float, default 60.0
        每个传感器的水平视场角 (0 < A ≤ 180)。
    center : {'auto', 'origin', ndarray}, default 'auto'
        角度判断的参考中心：
        * 'auto'   → 点云几何中心
        * 'origin' → (0,0,0)
        * ndarray  → 指定 (3,) 坐标
    return_masks : bool, default False
        若为 True，额外返回 shape=(num_sensors, N_pts) 的 bool 掩码矩阵。

    Returns
    -------
    list[np.ndarray] | (list[np.ndarray], np.ndarray)
        切分后的子点云列表，以及可选的掩码矩阵。
    """
    # --- 校验 ---------------------------------------------------------------
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是形状 (N_pts, 3) 的 ndarray")
    if not (0.0 < fov_deg <= 180.0):
        raise ValueError("fov_deg 必须位于 (0, 180]")

    # --- 参考中心 -----------------------------------------------------------
    if isinstance(center, str):
        center = center.lower()
        if center == "auto":
            c = points.mean(0)
        elif center == "origin":
            c = np.zeros(3, dtype=np.float64)
        else:
            raise ValueError("center 只能是 'auto'、'origin' 或 ndarray")
    else:
        c = np.asarray(center, dtype=np.float64)
        if c.shape != (3,):
            raise ValueError("center ndarray 必须形状为 (3,)")

    # --- 方向 & 单位向量 ----------------------------------------------------
    dirs = _sensor_directions(num_sensors)          # (N_s, 3)
    rel  = points - c                               # (N_pts, 3)
    rel_norm = np.linalg.norm(rel, axis=1, keepdims=True)
    rel_norm[rel_norm == 0] = 1e-8                  # 避免除零
    rel_unit = rel / rel_norm                       # (N_pts, 3)

    # --- 角度阈值 -----------------------------------------------------------
    cos_thresh = np.cos(np.deg2rad(fov_deg / 2.0))
    masks = (dirs @ rel_unit.T) >= cos_thresh       # (N_s, N_pts)

    slices = [points[m] for m in masks]
    return (slices, masks) if return_masks else slices
