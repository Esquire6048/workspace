import numpy as np

def inspect_npy(path: str) -> None:
    print(f"\n[检查文件] {path}")
    try:
        arr = np.load(path, allow_pickle=True)

        print(f"类型: {type(arr)}, 维度: {arr.ndim}, 形状: {arr.shape}, dtype: {arr.dtype}")

        # 尝试查看内容
        if arr.ndim == 0:
            # 可能是 dict 或其他对象
            val = arr.item()
            print(f"内部类型: {type(val)}")
            if isinstance(val, dict):
                print(f"🔍 是字典，包含键: {list(val.keys())}")
                for k in val:
                    print(f"  - {k}: shape = {val[k].shape}, dtype = {val[k].dtype}")
            else:
                print(f"📦 是标量对象：{val}")
        elif isinstance(arr, np.ndarray):
            print(f"✅ 是 ndarray，内容预览：{arr[:min(5, len(arr))]}")
    except Exception as e:
        print(f"❌ 读取失败：{e}")

# 示例用法
inspect_npy("./dataset/modelnet40_c/data_occlusion_2.npy")