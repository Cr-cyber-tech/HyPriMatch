from pathlib import Path
import numpy as np
import json
import argparse

P_DEFAULT = 2305843009213693951

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, default=r"E:\zyc\project\out\queries\q1",
                    help="query directory, e.g. ...\\queryset\\q0001")
    ap.add_argument("--p", type=int, default=P_DEFAULT, help="prime modulus")
    return ap.parse_args()

def colsum_mod_bigint(M, p):
    # fallback 用：兼容旧目录
    M = (M % p).astype(object)
    s = np.zeros((M.shape[1],), dtype=object)
    for j in range(M.shape[1]):
        s[j] = int(sum(M[:, j]) % p)
    return s.astype(np.int64)

def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)
    P = int(args.p)

    S = Q_DIR / "shares_dense"
    OUT = Q_DIR / "online_step12_cloud"
    OUT.mkdir(parents=True, exist_ok=True)

    # 优先直接读取预先算好的 size shares
    pre_sizeB_B = S / "sizeB_B.npy"
    pre_sizeQ_B = S / "sizeQ_B.npy"

    if pre_sizeB_B.exists() and pre_sizeQ_B.exists():
        sizeB_B = np.load(pre_sizeB_B)
        sizeQ_B = np.load(pre_sizeQ_B)
        mode = "precomputed_size_share"
    else:
        # fallback：兼容旧结果目录
        B_B = np.load(S / "B_B.npy")
        Q_B = np.load(S / "Q_B.npy")
        sizeB_B = colsum_mod_bigint(B_B, P)
        sizeQ_B = colsum_mod_bigint(Q_B, P)
        mode = "fallback_colsum_from_dense_share"

    np.save(OUT / "B_sizeB.npy", sizeB_B)
    np.save(OUT / "B_sizeQ.npy", sizeQ_B)

    meta = {
        "P": int(P),
        "E": int(sizeB_B.shape[0]),
        "Eq": int(sizeQ_B.shape[0]),
        "qdir": str(Q_DIR),
        "mode": mode
    }
    (OUT / "metaB.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[cloudB] mode =", mode)
    print("[cloudB] wrote:", OUT / "B_sizeB.npy", OUT / "B_sizeQ.npy")

if __name__ == "__main__":
    main()