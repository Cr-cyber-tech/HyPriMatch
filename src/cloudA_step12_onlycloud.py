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
    pre_sizeB_A = S / "sizeB_A.npy"
    pre_sizeQ_A = S / "sizeQ_A.npy"

    if pre_sizeB_A.exists() and pre_sizeQ_A.exists():
        sizeB_A = np.load(pre_sizeB_A)
        sizeQ_A = np.load(pre_sizeQ_A)
        mode = "precomputed_size_share"
    else:
        # fallback：兼容旧结果目录
        B_A = np.load(S / "B_A.npy")
        Q_A = np.load(S / "Q_A.npy")
        sizeB_A = colsum_mod_bigint(B_A, P)
        sizeQ_A = colsum_mod_bigint(Q_A, P)
        mode = "fallback_colsum_from_dense_share"

    np.save(OUT / "A_sizeB.npy", sizeB_A)
    np.save(OUT / "A_sizeQ.npy", sizeQ_A)

    meta = {
        "P": int(P),
        "E": int(sizeB_A.shape[0]),
        "Eq": int(sizeQ_A.shape[0]),
        "qdir": str(Q_DIR),
        "mode": mode
    }
    (OUT / "metaA.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[cloudA] mode =", mode)
    print("[cloudA] wrote:", OUT / "A_sizeB.npy", OUT / "A_sizeQ.npy")

if __name__ == "__main__":
    main()