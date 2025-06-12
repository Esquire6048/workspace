import os
import h5py
import numpy as np
from typing import List, Tuple


def read_file_list(txt_path: str) -> List[str]:
    """
    读取一个纯文本文件（例如 test_files.txt），
    读取每个非空行的文件路径
    返回文件路径列表
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"找不到文件列表：{txt_path}")
    
    paths = []

    with open(txt_path, 'r') as f:
        for line in f:
            p = line.strip()
            if p and p not in paths:
                paths.append(p)

    return paths


def load_h5_files(h5_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    传入 HDF5 文件路径列表
    依次加载每个文件里的 "data" 和 "label" 数据，
    返回合并后的点云数据 all_data 和标签 all_label
    """
    data_list = []
    label_list = []

    for h5_path in h5_paths:
        if not os.path.isfile(h5_path):
            print(f"[警告] 文件不存在，跳过：{h5_path}")
            continue

        with h5py.File(h5_path, 'r') as f:
            d = f['data'][:]
            l = f['label'][:]
            if l.ndim == 2 and l.shape[1] == 1:
                l = l.reshape(-1)
            data_list.append(d)
            label_list.append(l)
            print(f"已加载：{h5_path} → data: {d.shape}, label: {l.shape}")

    if not data_list:
        raise RuntimeError("未能加载到任何数据，请检查传入的 h5_paths 列表。")
    
    all_data = np.concatenate(data_list, axis=0)
    all_label = np.concatenate(label_list, axis=0)
    print(f"所有分片合并后：data shape = {all_data.shape}, label shape = {all_label.shape}")

    return all_data, all_label


def write_ply(points: np.ndarray, out_path: str) -> None:
    """
    将单个点云写为 PLY 格式（ASCII），仅包含顶点坐标，
    然后将 PLY 文件输出到指定路径
    """
    N = points.shape[0]

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w') as f:
        f.write("\n".join(header) + "\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"导出 PLY: {out_path}")