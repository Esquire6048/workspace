#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法
-----
# 处理单个 PLY
python ply_file_io.py ./dataset/LiDAR-Net/living/cloud_01_room01.ply

# 批量处理目录（会递归查找 *.ply）
python ply_file_io.py ./dataset/LiDAR-Net/living
"""
import sys, os, glob, copy, numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import open3d as o3d
from open3d import utility as o3du   # 仅为兼容旧代码，未显式使用

# ------------------- 可调参数 -------------------
HFOV_DEG     = 90        # 水平视角 (±45°)
VFOV_UP      = 15        # 向上角度
VFOV_DOWN    = 15        # 向下角度
R_MAX        = 20.0      # 最大量程
HPR_RADIUS   = 20.0      # HPR 球面半径 (通常与 R_MAX 相等)
VOXEL_SIZE   = 0.05      # 体素大小 (m)，0 表示不下采样
INSTALL_H    = 1.5       # 传感器安装高度 (m)
INWARD_SCALE = 0.8       # 角点向内缩比例
# ------------------------------------------------

# ------------------- 核心函数 -------------------
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
    # 传感器局部坐标系（+X 指向中心，水平朝向）
    vec_xy = center[:2] - sensor_pos[:2]
    yaw    = np.arctan2(vec_xy[1], vec_xy[0])
    R      = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    T      = np.eye(4); T[:3,:3], T[:3,3] = R, sensor_pos

    # 克隆全局点云 → 转到局部
    pcd_local = copy.deepcopy(pcd_global)
    pcd_local.transform(np.linalg.inv(T))
    attr = {k: ary.copy() for k, ary in attr_global.items()}

    # FOV 几何过滤
    pts = np.asarray(pcd_local.points)
    dx, dy, dz = pts[:,0], pts[:,1], pts[:,2]
    rr = np.linalg.norm(pts, axis=1)
    th = np.degrees(np.arctan2(dy, dx))
    ph = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))
    sel_idx = np.where( (np.abs(th)<hfov/2) &
                        (ph<v_up) & (ph>-v_down) &
                        (rr<r_max) )[0]
    pcd_vis = pcd_local.select_by_index(sel_idx)
    for k in attr:
        attr[k] = attr[k][sel_idx]

    # 体素下采样（首点）
    if voxel and voxel > 0 and len(pcd_vis.points):
        _, trace, _ = pcd_vis.voxel_down_sample_and_trace(
            voxel_size=voxel,
            min_bound=pcd_vis.get_min_bound() - 1e-3,
            max_bound=pcd_vis.get_max_bound() + 1e-3,
            approximate_class=False)

        first_idx = [int(lst[0]) for lst in trace if len(lst)]

        # --- NumPy 取坐标、重建 PointCloud ---
        pts_keep = np.asarray(pcd_vis.points)[first_idx]
        pcd_vis  = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(pts_keep))

        # --- 同步裁剪属性 ---
        attr = {k: ary[first_idx] for k, ary in attr.items()}

    # 变回全局
    pcd_vis.transform(T)
    return pcd_vis, attr

def inspect_ply(path):
    """读取单个 PLY 文件，裁剪四视角子云并保存"""
    basename = os.path.splitext(os.path.basename(path))[0]

    # 1) 读取 PLY, 打印 header & 示例
    with open(path, 'rb') as f:
        header = []
        while True:
            l = f.readline().decode('ascii', 'ignore').rstrip()
            header.append(l)
            if l == 'end_header': break
    print(f"\n=== {basename}: PLY 头部 (前几行) ===")
    for l in header[:10]: print(l)
    if len(header) > 10: print("...")

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

    # 2) AABB & 传感器位置
    xmin,xmax = xs.min(), xs.max()
    ymin,ymax = ys.min(), ys.max()
    zmin,zmax = zs.min(), zs.max()

    xy = np.vstack([xs, ys]).T
    hull = ConvexHull(xy)
    poly = Polygon(xy[hull.vertices])
    rect4 = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:-1]
    ctr   = rect4.mean(axis=0)
    inner4 = ctr + INWARD_SCALE * (rect4 - ctr)      # 向内缩
    z_floor = zmin
    sensors = np.column_stack([inner4,
                               np.full(4, z_floor + INSTALL_H)])

    # 3) 准备 Open3D 点云
    pts = np.vstack([xs, ys, zs]).T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    center = np.array([xs.mean(), ys.mean(), z_floor + INSTALL_H])

    # 4) 对每台 LiDAR 裁剪视角子云并写文件
    for idx, pos in enumerate(sensors):
        pcd_sub, attr_sub = simulate_lidar_view(pcd, attr_global, pos, center)
        n = len(pcd_sub.points)
        print(f"[{basename}] Sensor {idx}: 子云点数 {n}")
        if n == 0:
            continue

        xyz = np.asarray(pcd_sub.points, dtype=np.float32)
        rec = np.empty(n, dtype=[
            ('x','f4'), ('y','f4'), ('z','f4'),
            ('red','u1'), ('green','u1'), ('blue','u1'),
            ('sem','i4'), ('ins','i4')
        ])
        rec['x'], rec['y'], rec['z'] = xyz.T
        rec['red']   = attr_sub['red']
        rec['green'] = attr_sub['green']
        rec['blue']  = attr_sub['blue']
        rec['sem']   = attr_sub['sem']
        rec['ins']   = attr_sub['ins']

        out_dir = "sliced_views"
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.join(out_dir, f"{basename}_lidar{idx}.ply")
        PlyData([PlyElement.describe(rec, 'vertex')], text=False).write(out_name)

# ------------------- 主入口 -------------------
def all_ply_files(target_path):
    """返回 target_path 下所有 ply 路径（递归）"""
    if os.path.isfile(target_path):
        return [target_path]
    # is dir
    pattern = os.path.join(target_path, "**", "*.ply")
    return glob.glob(pattern, recursive=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    ply_list = all_ply_files(target)
    if not ply_list:
        print("未找到任何 .ply 文件")
        sys.exit(1)

    for p in ply_list:
        try:
            inspect_ply(p)
        except Exception as e:
            print(f"[Error] 处理 {p} 时出错：{e}")
