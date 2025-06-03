import h5py
import numpy as np

with h5py.File('ply_data_train0.h5', 'r') as f:
    # 查看整个层次结构（可迭代打印组与数据集）
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")
    f.visititems(print_structure)

    # 读取 train/points 部分（但不一次性全部加载到内存）
    # 例如只读取第 0~99 条样本
    train_pts_dset = f['data/train/points']
    chunk0_100 = train_pts_dset[0:10]  # shape = (100, 1024, 3), numpy array

    # 读取对应标签
    train_lbls = f['data/train/labels'][0:10]  # shape = (100,)
    
    # 读取整个测试集
    test_pts = f['data/test/points'][:]  # shape = (N_test, 1024, 3)
    test_lbl = f['data/test/labels'][:]  # shape = (N_test,)
    
    print("前 5 条训练标签：", train_lbls[:5])