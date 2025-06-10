import os
import h5py
import argparse
import numpy as np
from typing import List, Tuple, Dict

def read_file_list(txt_path: str) -> List[str]:
    """
    读取一个纯文本文件（例如 test_files.txt），
    返回每一行去掉空白和换行符后得到的文件路径列表。
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
    根据给定的 HDF5 文件路径列表，依次加载每个文件里的 "data" 和 "label" 数据，
    返回合并后的点云数据 all_data 和标签 all_label。
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
    然后将 PLY 文件输出到指定目录。
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


def export_sample_slices(
    pts: np.ndarray,
    label: int,
    idx: int,
    num_slices: int,
    out_dir: str
) -> None:
    """
    对单个样本的点云进行方位角切片并导出 PLY 文件。
    样本索引 idx 和标签 label 会被用于生成文件名。
    """
    if num_slices > 1:
        centered = pts - pts.mean(axis=0)
        thetas = np.degrees(np.arctan2(centered[:,1], centered[:,0]))
        thetas = np.mod(thetas, 360)
        span = 360.0 / num_slices
        for s in range(num_slices):
            mask = (thetas >= s * span) & (thetas < (s + 1) * span)
            slice_pts = pts[mask]
            if slice_pts.size == 0:
                print(f"[警告] 样本 {idx} 第 {s} 扇区无点，跳过。")
                continue
            filename = f"sample_{idx:04d}_slice{s:02d}_cls{label}.ply"
            out_path = os.path.join(out_dir, filename)
            write_ply(slice_pts, out_path)
    else:
        filename = f"sample_{idx:04d}_cls{label}.ply"
        out_path = os.path.join(out_dir, filename)
        write_ply(pts, out_path) 


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 ModelNet40 HDF5 导出点云到 PLY，并可按方位切片"
    )
    parser.add_argument('--file_list', type=str, required=True,
                        help='HDF5 文件列表路径，每行一个 .h5 文件')
    parser.add_argument('--sample_idxs', type=int, nargs='+', default=[0],
                        help='要导出的样本索引列表，基于合并后的数组')
    parser.add_argument('--out_dir', type=str, default='output_ply',
                        help='导出 PLY 文件保存目录')
    parser.add_argument('--slices', type=int, default=1,
                        help='要切分的方位扇区数量 N，默认为1（不切分）')
    return parser.parse_args()


def main():
    args = parse_args()

    # —— 1. 读取 file_list.txt，并加载所有 HDF5 —— #
    h5_paths = read_file_list(args.file_list)

    print("==> 即将加载的 HDF5 文件列表：")
    for p in h5_paths:
        print("    ", p)

    data, labels = load_h5_files(h5_paths)

    # —— 2. 导出指定样本 ===== #
    N = args.slices
    for idx in args.sample_idxs:
        if idx < 0 or idx >= data.shape[0]:
            print(f"[警告] 索引 {idx} 超出范围 (0 to {data.shape[0]-1})，跳过。")
            continue
        pts = data[idx]
        label = labels[idx]
        export_sample_slices(pts, int(label), idx, N, args.out_dir)


if __name__ == "__main__":
    main()