# 用法:  python inspect_ply_and_slice.py path/to/room.ply

import sys, numpy as np
import copy
from plyfile import PlyElement, PlyData
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

import open3d as o3d      # 只做几何运算，无 GUI

# ------------------------------ 可调参数 ------------------------------
HFOV_DEG     = 90          # 水平视角  (deg)     例：90° = ±45°
VFOV_UP      = 15          # 向上角度  (deg)
VFOV_DOWN    = 15          # 向下角度  (deg)
R_MAX        = 20.0        # 最大量程 (m)
HPR_RADIUS   = 20.0        # HPR 球面半径 (和 R_MAX 一般相等即可)
VOXEL_SIZE   = 0.05        # 下采样体素，为 0 跳过下采样
INSTALL_H    = 1.5         # 传感器距地高度 (m)
INWARD_SCALE = 0.8        # 角点向内缩比例 (0.95 = 缩 5 %)
# ---------------------------------------------------------------------

def simulate_lidar_view(pcd_global,
                        attr_global,
                        sensor_pos,
                        center,
                        hfov=HFOV_DEG,
                        v_up=VFOV_UP,
                        v_down=VFOV_DOWN,
                        r_max=R_MAX,
                        r_hpr=HPR_RADIUS,
                        voxel=VOXEL_SIZE):
    """
    仅按视场 (FOV) 裁剪，不再做遮挡判定。
    返回：裁剪后的子云（仍在全局坐标系下）
    """
    # ① 构造局部坐标系（让 +X 指向中心）
    vec_xy = center[:2] - sensor_pos[:2]
    yaw = np.arctan2(vec_xy[1], vec_xy[0])
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, sensor_pos

    # ② 全局点云 → 局部坐标
    pcd_local = copy.deepcopy(pcd_global)
    pcd_local.transform(np.linalg.inv(T))
    attr = {k: ary.copy() for k, ary in attr_global.items()}

    # ③ 纯 FOV 几何过滤（扫描锥体）
    pts = np.asarray(pcd_local.points)
    dx, dy, dz = pts[:, 0], pts[:, 1], pts[:, 2]
    rr = np.linalg.norm(pts, axis=1)
    th = np.degrees(np.arctan2(dy, dx))                    # 水平角
    ph = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))# 垂直角

    sel_idx = np.where( (np.abs(th)<hfov/2) &
                        (ph<v_up) & (ph>-v_down) &
                        (rr<r_max) )[0]

    pcd_vis = pcd_local.select_by_index(sel_idx)

    attr = {k: ary[sel_idx] for k, ary in attr_global.items()} 

    # ⑤ 再变回全局坐标
    pcd_vis.transform(T)
    return pcd_vis, attr

# ---------------------------------------------------------------------
def inspect_ply(path):
    # 1) 读取 PLY, 打印 header & 示例
    with open(path, 'rb') as f:
        header = []
        while True:
            l = f.readline().decode('ascii', 'ignore').rstrip()
            header.append(l)
            if l == 'end_header': break
    print("=== PLY Header ===\n", "\n".join(header), "\n")

    ply = PlyData.read(path)
    v   = ply['vertex'].data
    xs, ys, zs = v['x'], v['y'], v['z']

    attr_global = {
        "red":   v['red'].astype(np.uint8),
        "green": v['green'].astype(np.uint8),
        "blue":  v['blue'].astype(np.uint8),
        "sem":   v['sem'].astype(np.int32),
        "ins":   v['ins'].astype(np.int32),
    }

    print("=== Example record ===\n", {n:v[0][n] for n in v.dtype.names}, "\n")

    # 2) AABB
    xmin,xmax = xs.min(), xs.max()
    ymin,ymax = ys.min(), ys.max()
    zmin,zmax = zs.min(), zs.max()
    print("=== AABB ===")
    print(f"x:[{xmin:.3f},{xmax:.3f}] Δ={xmax-xmin:.3f}")
    print(f"y:[{ymin:.3f},{ymax:.3f}] Δ={ymax-ymin:.3f}")
    print(f"z:[{zmin:.3f},{zmax:.3f}] Δ={zmax-zmin:.3f}\n")

    # 3) 找四角传感器位置 (凸包→MBR→缩 inward_scale)
    xy = np.vstack([xs, ys]).T
    hull = ConvexHull(xy)
    poly = Polygon(xy[hull.vertices])
    rect4 = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:-1]
    ctr   = rect4.mean(axis=0)
    inner4 = ctr + INWARD_SCALE * (rect4 - ctr)  # 向内缩
    z_floor = zmin
    sensors = np.column_stack([inner4,
                               np.full(4, z_floor + INSTALL_H)])
    print("≈ 4  个 LiDAR 安装点:")
    for i,p in enumerate(sensors): print(f"{i}: {p}")

    # 4) 准备 Open3D 点云
    pts = np.vstack([xs, ys, zs]).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    center = np.array([xs.mean(), ys.mean(), z_floor + INSTALL_H])

    # 5) 对每台 LiDAR 裁剪视角子云
    for idx, pos in enumerate(sensors):
        pcd_sub, attr_sub = simulate_lidar_view(pcd, attr_global, pos, center)
        n = len(pcd_sub.points)
        print(f"[Sensor {idx}] 子云点数: {n}")
        if n == 0:
            continue

        xyz = np.asarray(pcd_sub.points, dtype=np.float32)
        rec = np.empty(n, dtype=[
            ('x','f4'),
            ('y','f4'),
            ('z','f4'),
            ('red','u1'),
            ('green','u1'),
            ('blue','u1'),
            ('sem','i4'),
            ('ins','i4')
        ])
        rec['x'], rec['y'], rec['z'] = xyz.T
        rec['red']   = attr_sub['red']
        rec['green'] = attr_sub['green']
        rec['blue']  = attr_sub['blue']
        rec['sem']   = attr_sub['sem']
        rec['ins']   = attr_sub['ins']

        PlyData([PlyElement.describe(rec, 'vertex')],
                text=False).write(f"lidar{idx}_view.ply")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_ply_and_slice.py path/to/room.ply")
        sys.exit(1)
    inspect_ply(sys.argv[1])
