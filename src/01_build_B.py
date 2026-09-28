from pathlib import Path
import json
import numpy as np
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm
import argparse

# === 默认路径：保持你现在 HC 的行为不变 ===
HYPEREDGES_PATH_DEFAULT = r"E:\zyc\project\data\hc\hyperedges-house-committees.txt"
NODE_NAMES_PATH_DEFAULT = r"E:\zyc\project\data\hc\node-names-house-committees.txt"
OUT_DIR_DEFAULT = r"E:\zyc\project\out"
OUT_NPZ_NAME_DEFAULT = "hc_incidence_VxE.npz"
OUT_META_NAME_DEFAULT = "hc_meta.json"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyperedges_path", type=str, default=HYPEREDGES_PATH_DEFAULT,
                    help="path to hyperedges txt")
    ap.add_argument("--node_names_path", type=str, default=NODE_NAMES_PATH_DEFAULT,
                    help="path to node names txt")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT,
                    help="output directory")
    ap.add_argument("--out_npz_name", type=str, default=OUT_NPZ_NAME_DEFAULT,
                    help="output incidence npz filename")
    ap.add_argument("--out_meta_name", type=str, default=OUT_META_NAME_DEFAULT,
                    help="output meta json filename")
    return ap.parse_args()

def load_lines(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()

def determine_index_base(hyperedges_path: Path, node_count: int):
    """更智能地确定索引是0-based还是1-based"""
    print("正在分析索引编号规则...")

    edge_lines = load_lines(hyperedges_path)
    all_indices = []

    for line in edge_lines[:min(100, len(edge_lines))]:  # 检查前100行
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        for p in parts:
            try:
                all_indices.append(int(p))
            except ValueError:
                pass

    if not all_indices:
        raise ValueError("无法从数据中提取索引")

    min_idx = min(all_indices)
    max_idx = max(all_indices)

    print(f"  节点总数: {node_count}")
    print(f"  样本最小索引: {min_idx}")
    print(f"  样本最大索引: {max_idx}")
    print(f"  样本索引范围: {max_idx - min_idx + 1}")

    # 判断规则
    if min_idx == 0:
        print("  → 检测到 0-based 索引 (索引从0开始)")
        return 0  # 0-based
    elif min_idx == 1 and max_idx == node_count:
        print(f"  → 检测到 1-based 索引 (索引从1开始，最大值等于节点数 {node_count})")
        return -1  # 1-based，需要减1
    elif max_idx < node_count:
        print(f"  → 可能是 0-based 索引 (最大值 {max_idx} < 节点数 {node_count})")
        return 0
    elif max_idx == node_count:
        print(f"  → 可能是 1-based 索引 (最大值 {max_idx} == 节点数 {node_count})")
        return -1
    else:
        print(f"  → 无法确定索引规则，尝试 0-based")
        return 0

def build_incidence_matrix(hyperedges_path: Path, node_names_path: Path):
    if not hyperedges_path.exists():
        raise FileNotFoundError(f"找不到超边文件: {hyperedges_path}")
    if not node_names_path.exists():
        raise FileNotFoundError(f"找不到节点文件: {node_names_path}")

    # 读取节点名称
    node_names = load_lines(node_names_path)
    V = len(node_names)
    print(f"读取到 {V} 个节点")

    # 确定索引偏移量
    index_shift = determine_index_base(hyperedges_path, V)

    # 读取超边
    edge_lines = load_lines(hyperedges_path)
    E = len(edge_lines)
    print(f"读取到 {E} 条超边")

    rows, cols = [], []
    problematic_lines = []

    for e, line in tqdm(list(enumerate(edge_lines)), desc="解析超边"):
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        nodes = []
        line_valid = True

        for p in parts:
            try:
                v_raw = int(p)
                v = v_raw + index_shift  # 应用偏移

                if v < 0 or v >= V:
                    problematic_lines.append({
                        "line_num": e + 1,
                        "raw_value": v_raw,
                        "adjusted_value": v,
                        "max_allowed": V - 1
                    })
                    line_valid = False
                    continue

                nodes.append(v)
            except ValueError:
                print(f"警告: 第 {e+1} 行包含非整数: {p}")
                line_valid = False

        if line_valid and nodes:
            # 去重并排序（保留你原逻辑）
            nodes = sorted(set(nodes))
            rows.extend(nodes)
            cols.extend([e] * len(nodes))
        elif not line_valid:
            print(f"警告: 第 {e+1} 行包含无效数据，已跳过")

    # 检查是否有问题
    if problematic_lines:
        print(f"\n警告: 发现 {len(problematic_lines)} 行数据有问题")
        for i, prob in enumerate(problematic_lines[:5]):  # 只显示前5个问题
            print(f"  第 {prob['line_num']} 行: raw={prob['raw_value']}, "
                  f"adjusted={prob['adjusted_value']}, allowed=0-{prob['max_allowed']}")

        if len(problematic_lines) > 5:
            print(f"  ... 还有 {len(problematic_lines) - 5} 个类似问题")

        # 尝试自动修复：如果大多数问题是索引过大，可能是1-based
        if index_shift == 0:
            large_indices = [p for p in problematic_lines if p["raw_value"] >= V]
            if len(large_indices) > len(problematic_lines) * 0.5:  # 超过50%的问题
                print("\n检测到大量索引越界，尝试使用 1-based 索引...")
                index_shift = -1
                print(f"  重新解析数据，使用偏移量 {index_shift}")

                # 重新解析
                rows, cols = [], []
                problematic_lines = []

                for e, line in tqdm(list(enumerate(edge_lines)), desc="重新解析超边"):
                    line = line.strip()
                    if not line:
                        continue

                    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                    nodes = []

                    for p in parts:
                        try:
                            v_raw = int(p)
                            v = v_raw + index_shift

                            if v < 0 or v >= V:
                                problematic_lines.append({
                                    "line_num": e + 1,
                                    "raw_value": v_raw,
                                    "adjusted_value": v,
                                    "max_allowed": V - 1
                                })
                                continue

                            nodes.append(v)
                        except ValueError:
                            continue

                    if nodes:
                        nodes = sorted(set(nodes))
                        rows.extend(nodes)
                        cols.extend([e] * len(nodes))

    if problematic_lines:
        print(f"\n错误: 仍有 {len(problematic_lines)} 行数据无法处理")
        print("请检查数据文件格式是否正确")
        raise ValueError(f"发现 {len(problematic_lines)} 行无效数据")

    # 创建稀疏矩阵
    data = np.ones(len(rows), dtype=np.uint8)
    B = coo_matrix((data, (rows, cols)), shape=(V, E)).tocsr()

    return B, V, E

def main():
    args = parse_args()

    hyperedges_path = Path(args.hyperedges_path)
    node_names_path = Path(args.node_names_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / args.out_npz_name
    out_meta = out_dir / args.out_meta_name

    print("从以下文件构建关联矩阵 B:")
    print(f"  超边文件: {hyperedges_path}")
    print(f"  节点文件: {node_names_path}")

    try:
        B, V, E = build_incidence_matrix(hyperedges_path, node_names_path)
    except Exception as e:
        print(f"\n构建矩阵时出错: {e}")

        # 提供调试信息
        print("\n调试信息:")
        print("1. 检查节点数量:")
        if node_names_path.exists():
            with node_names_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"  节点文件行数: {len(lines)}")
                if len(lines) > 0:
                    print(f"  第一行: {lines[0].strip()}")
                if len(lines) > 1:
                    print(f"  最后一行: {lines[-1].strip()}")

        print("\n2. 检查超边文件格式:")
        if hyperedges_path.exists():
            with hyperedges_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"  总行数: {len(lines)}")
                for i in range(max(0, len(lines) - 5), len(lines)):
                    print(f"  第 {i+1} 行: {lines[i].strip()}")
        return

    print(f"\n构建完成!")
    print(f"矩阵 B 的形状 (V x E) = {B.shape}")
    print(f"非零元素数量 = {B.nnz}")

    # 计算统计信息
    edge_sizes = np.asarray(B.sum(axis=0)).ravel()
    node_degrees = np.asarray(B.sum(axis=1)).ravel()

    stats = {
        "V": int(V),
        "E": int(E),
        "nnz": int(B.nnz),
        "edge_size_min": int(edge_sizes.min()) if E > 0 else 0,
        "edge_size_avg": float(edge_sizes.mean()) if E > 0 else 0.0,
        "edge_size_max": int(edge_sizes.max()) if E > 0 else 0,
        "node_degree_min": int(node_degrees.min()) if V > 0 else 0,
        "node_degree_avg": float(node_degrees.mean()) if V > 0 else 0.0,
        "node_degree_max": int(node_degrees.max()) if V > 0 else 0,
        "hyperedges_path": str(hyperedges_path),
        "node_names_path": str(node_names_path),
    }

    # 保存文件
    save_npz(out_npz, B)

    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n文件已保存:")
    print(f"  矩阵文件: {out_npz}")
    print(f"  元数据: {out_meta}")

    print("\n统计信息:")
    print(f"  超边大小: 最小={stats['edge_size_min']}, "
          f"平均={stats['edge_size_avg']:.2f}, "
          f"最大={stats['edge_size_max']}")
    print(f"  节点度: 最小={stats['node_degree_min']}, "
          f"平均={stats['node_degree_avg']:.2f}, "
          f"最大={stats['node_degree_max']}")

if __name__ == "__main__":
    main()