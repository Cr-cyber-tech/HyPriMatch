from pathlib import Path
import json
import numpy as np
from scipy.sparse import load_npz
import argparse

B_PATH_DEFAULT = r"E:\zyc\project\out\hc_incidence_VxE.npz"
Q_DIR_DEFAULT  = r"E:\zyc\project\out\queries\q1"

P_DEFAULT    = 2305843009213693951
SEED_DEFAULT = 20260121

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, default=Q_DIR_DEFAULT,
                    help="query dir, e.g. D:\\hypergraph\\project\\out\\queries\\queryset\\q0008")
    ap.add_argument("--bpath", type=str, default=B_PATH_DEFAULT,
                    help="incidence matrix B path (.npz)")
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--global_data_share_dir", type=str, default=None,
                    help="global dir to store shared data-hypergraph tensors (B_A/B_B/sizeB_A/sizeB_B)")
    return ap.parse_args()

def share_modp(X, p, rng):
    """
    加法秘密共享（mod p）
    X 可以是向量，也可以是矩阵
    """
    X = X.astype(np.int64) % p
    A = rng.integers(0, p, size=X.shape, dtype=np.int64)
    B = (X - A) % p
    return A, B

def main():
    args = parse_args()
    print("[DEBUG] argv qdir =", args.qdir)

    B_PATH = Path(args.bpath)
    Q_DIR = Path(args.qdir)
    print("[DEBUG] using Q_DIR =", Q_DIR)
    print("[DEBUG] using B_PATH =", B_PATH)

    global_data_share_dir = Path(args.global_data_share_dir) if args.global_data_share_dir else None
    if global_data_share_dir is None:
        raise ValueError("--global_data_share_dir is required")
    global_data_share_dir.mkdir(parents=True, exist_ok=True)
    print("[DEBUG] using global_data_share_dir =", global_data_share_dir)

    Q_PATH = Q_DIR / "Q_incidence_UxEq.npz"
    INTER_PATH = Q_DIR / "Q_intersections_EqxEq.npy"

    if not Q_PATH.exists():
        raise FileNotFoundError(f"Missing {Q_PATH}")
    if not INTER_PATH.exists():
        raise FileNotFoundError(f"Missing {INTER_PATH}")
    if not B_PATH.exists():
        raise FileNotFoundError(f"Missing {B_PATH}")

    OUT_DIR = Q_DIR / "shares_dense"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    P = int(args.p)
    SEED = int(args.seed)
    rng = np.random.default_rng(SEED)

    # ===== 全局数据 share 文件路径 =====
    GA_B_A = global_data_share_dir / "B_A.npy"
    GA_B_B = global_data_share_dir / "B_B.npy"
    GA_sizeB_A = global_data_share_dir / "sizeB_A.npy"
    GA_sizeB_B = global_data_share_dir / "sizeB_B.npy"
    GA_meta = global_data_share_dir / "data_share_meta.json"

    # ===== 数据超图 B：全局只生成一次 =====
    need_build_global_B = not (
        GA_B_A.exists() and GA_B_B.exists() and GA_sizeB_A.exists() and GA_sizeB_B.exists()
    )

    if need_build_global_B:
        print("[GLOBAL] building data shares ...")
        B_sp = load_npz(B_PATH).tocsr()
        B_shape = list(B_sp.shape)
        B_nnz = int(B_sp.nnz)

        # 明文 sizeB 只在内存中短暂存在；不落盘
        sizeB = np.asarray(B_sp.sum(axis=0)).ravel().astype(np.int64)
        sizeB_A, sizeB_B = share_modp(sizeB, P, rng)
        del sizeB

        # 为 Step2 / Step3 准备全局数据矩阵 share
        B = B_sp.toarray().astype(np.int64)
        B_A, B_B = share_modp(B, P, rng)
        del B
        del B_sp

        np.save(GA_B_A, B_A)
        np.save(GA_B_B, B_B)
        np.save(GA_sizeB_A, sizeB_A)
        np.save(GA_sizeB_B, sizeB_B)

        # 释放大对象
        del B_A
        del B_B
        del sizeB_A
        del sizeB_B

        GA_meta.write_text(
            json.dumps({
                "P": int(P),
                "SEED": int(SEED),
                "B_shape": B_shape,
                "B_nnz": B_nnz,
                "note": "plain sizeB is computed in memory only and never written to disk"
            }, indent=2),
            encoding="utf-8"
        )

        global_mode = "built_now"
        print("[GLOBAL] saved:", GA_B_A, GA_B_B, GA_sizeB_A, GA_sizeB_B)
    else:
        print("[GLOBAL] reuse existing data shares from:", global_data_share_dir)
        if GA_meta.exists():
            meta_obj = json.loads(GA_meta.read_text(encoding="utf-8"))
            B_shape = meta_obj.get("B_shape", list(load_npz(B_PATH).shape))
            B_nnz = int(meta_obj.get("B_nnz", 0))
        else:
            B_sp_tmp = load_npz(B_PATH)
            B_shape = list(B_sp_tmp.shape)
            B_nnz = int(B_sp_tmp.nnz)
            del B_sp_tmp
        global_mode = "reused_existing"

    # ===== 查询子图 Q / inter：每个 query 单独生成 =====
    Q_sp = load_npz(Q_PATH).tocsr()

    
    sizeQ = np.asarray(Q_sp.sum(axis=0)).ravel().astype(np.int64)
    sizeQ_A, sizeQ_B = share_modp(sizeQ, P, rng)
    del sizeQ

    Q = Q_sp.toarray().astype(np.int64)
    Q_shape = list(Q.shape)
    Q_nnz = int(Q.sum())
    Q_A, Q_B = share_modp(Q, P, rng)
    del Q
    del Q_sp

    inter = np.load(INTER_PATH).astype(np.int64)
    np.fill_diagonal(inter, 0)
    inter_shape = list(inter.shape)
    inter_A, inter_B = share_modp(inter, P, rng)
    del inter

    # ===== 保存当前 query 的 share =====
    np.save(OUT_DIR / "Q_A.npy", Q_A)
    np.save(OUT_DIR / "Q_B.npy", Q_B)
    np.save(OUT_DIR / "inter_A.npy", inter_A)
    np.save(OUT_DIR / "inter_B.npy", inter_B)
    np.save(OUT_DIR / "sizeQ_A.npy", sizeQ_A)
    np.save(OUT_DIR / "sizeQ_B.npy", sizeQ_B)

    # 为兼容当前 Step12 onlycloud，给当前 query 放一份小的 sizeB share 副本
    sizeB_A_local = np.load(GA_sizeB_A)
    sizeB_B_local = np.load(GA_sizeB_B)
    np.save(OUT_DIR / "sizeB_A.npy", sizeB_A_local)
    np.save(OUT_DIR / "sizeB_B.npy", sizeB_B_local)
    del sizeB_A_local
    del sizeB_B_local

    # 释放 query share 对象
    del Q_A
    del Q_B
    del inter_A
    del inter_B
    del sizeQ_A
    del sizeQ_B

    params = {
        "P": int(P),
        "SEED": int(SEED),
        "global_data_share_dir": str(global_data_share_dir),
        "global_mode": global_mode,
        "B_shape": B_shape,
        "Q_shape": Q_shape,
        "inter_shape": inter_shape,
        "B_nnz": int(B_nnz),
        "Q_nnz": int(Q_nnz),
        "sizeB_shape": list(np.load(GA_sizeB_A, mmap_mode="r").shape),
        "sizeQ_shape": list(np.load(OUT_DIR / "sizeQ_A.npy", mmap_mode="r").shape),
        "note": "plain size vectors are computed in memory only and never written to disk; B_A/B_B are stored globally, not per-query"
    }
    (OUT_DIR / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    print("Saved query shares to:", OUT_DIR)
    print("Global data shares at:", global_data_share_dir)

if __name__ == "__main__":
    main()