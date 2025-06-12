# point_cloud_slice.py
"""
核心函数 `slice_point_cloud()`
==============================

根据 **N 个虚拟摄像机** 与 **水平视场角 (conical FOV)**，
将单个 `(N_pts,3)` 点云切分为 `N` 个子点云，
模拟多视角观测。

本版本假设 **XZ 平面为水平面**，
Y 轴为高度。

用法示例
--------
```python
from point_cloud_slice import slice_point_cloud
sub_clouds = slice_point_cloud(
    points,
    num_sensors=4,
    fov_deg=45,
    center='auto',
    sensor_distance=None
)
```

返回
----
- `slices`: 长度 = `num_sensors` 的子点云列表
- 可选 `masks`: 如果 `return_masks=True`，则一并返回 `(num_sensors, N_pts)` 布尔矩阵
"""
from typing import List, Tuple, Union, Optional
import numpy as np

__all__ = ["slice_point_cloud"]

def slice_point_cloud(
    points: np.ndarray,
    *,
    num_sensors: int = 3,
    fov_deg: float = 60.0,
    center: Union[str, np.ndarray] = "auto",
    sensor_distance: Optional[float] = None,
    return_masks: bool = False,
) -> List[np.ndarray] | Tuple[List[np.ndarray], np.ndarray]:
    """
    按锥形视场 (conical FOV) 切分点云。

    假设 XZ 平面为水平面，Y 轴为高度。
    """
    # 校验
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是形状 (N_pts,3) 的 ndarray")
    if not (0.0 < fov_deg <= 180.0):
        raise ValueError("fov_deg 必须位于 (0,180] 区间")

    # 参考中心
    if isinstance(center, str):
        cstr = center.lower()
        if cstr == 'auto':
            c = points.mean(axis=0)
        elif cstr == 'origin':
            c = np.zeros(3, dtype=float)
        else:
            raise ValueError("center 只能是 'auto'、'origin' 或 ndarray")
    else:
        c = np.asarray(center, dtype=float)
        if c.shape != (3,):
            raise ValueError("center ndarray 必须形状为 (3,)")

    # 计算水平最大半径：XZ 平面
    horiz = points[:, [0, 2]] - c[[0, 2]]  # XZ
    dists = np.linalg.norm(horiz, axis=1)
    R = float(sensor_distance) if sensor_distance is not None else float(dists.max())

    # 构造传感器位置：在 XZ 平面圆周均匀分布，Y=中心高度
    az = np.linspace(0.0, 2.0*np.pi, num_sensors, endpoint=False)
    sx = c[0] + R * np.cos(az)
    sz = c[2] + R * np.sin(az)
    sy = np.full(num_sensors, c[1])
    sensor_pos = np.stack([sx, sy, sz], axis=1)  # (num_sensors,3)

    # 方向向量：指向中心，单位化
    dirs = c - sensor_pos                       # (num_sensors,3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    # 计算每个 sensor 到每个点的单位向量
    # rel shape = (num_sensors, N_pts, 3)
    rel = points[None, :, :] - sensor_pos[:, None, :]
    rel_norm = np.linalg.norm(rel, axis=2, keepdims=True)
    rel_norm[rel_norm == 0] = 1e-8
    rel_unit = rel / rel_norm

    # 锥形 FOV 判定
    cos_th = np.cos(np.deg2rad(fov_deg/2.0))
    # 点积：dirs (s,3) 与 rel_unit (s, N_pts,3)
    dots = np.einsum('sd,spd->sp', dirs, rel_unit)  # (num_sensors, N_pts)
    masks = dots >= cos_th

    # 输出切分
    slices = [points[masks[i]] for i in range(num_sensors)]
    return (slices, masks) if return_masks else slices
