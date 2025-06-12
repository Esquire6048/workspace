import os
import h5py
import argparse
import numpy as np
from typing import List, Tuple, Dict
from h5_file_io import read_file_list, load_h5_files, write_ply


def export_sample_slices(
    pts: np.ndarray,
    label: int,
    idx: int,
    out_dir: str,
    num_slices: int = 3,
    fov_deg: float = 60.0,
) -> None:
    """
    对单个样本的点云进行方位角切片并导出 PLY 文件。
    样本索引 idx 和标签 label 会被用于生成文件名。
    """
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