from pathlib import Path
import json
import numpy as np
from scipy.sparse import load_npz
import argparse

B_PATH_DEFAULT = r"E:\zyc\project\out_ma\ma_incidence_VxE.npz"
OUT_DIR_DEFAULT = r"E:\zyc\project\out_ma\data_shares"

P_DEFAULT = 2305843009213693951
SEED_DEFAULT = 20260401

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpath", type=str, default=B_PATH_DEFAULT,
                    help="incidence matrix B path (.npz), owner-side plaintext hypergraph incidence")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT,
                    help="global data share dir, e.g. D:\\hypergraph\\project\\out_ma\\data_shares")
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)

    # 默认复用已有 B_A/B_B/sizeB_A/sizeB_B，只补 C_A/C_B
    ap.add_argument("--rebuild_b", action="store_true",
                    help="rebuild B_A/B_B and sizeB_A/sizeB_B even if they already exist")
    ap.add_argument("--rebuild_c", action="store_true",
                    help="rebuild C_A/C_B even if they already exist")
    return ap.parse_args()

def share_modp(X, p, rng):
    """
    Additive secret sharing over mod p.
    X can be vector or matrix.
    """
    X = X.astype(np.int64) % p
    A = rng.integers(0, p, size=X.shape, dtype=np.int64)
    B = (X - A) % p
    return A, B

def main():
    args = parse_args()

    B_PATH = Path(args.bpath)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not B_PATH.exists():
        raise FileNotFoundError(f"Missing {B_PATH}")

    P = int(args.p)
    SEED = int(args.seed)
    rng = np.random.default_rng(SEED)

    # global output files
    B_A_path = OUT_DIR / "B_A.npy"
    B_B_path = OUT_DIR / "B_B.npy"
    sizeB_A_path = OUT_DIR / "sizeB_A.npy"
    sizeB_B_path = OUT_DIR / "sizeB_B.npy"
    C_A_path = OUT_DIR / "C_A.npy"
    C_B_path = OUT_DIR / "C_B.npy"
    meta_path = OUT_DIR / "data_share_meta.json"

    # read current meta if exists
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    print("[OWNER03] using B_PATH =", B_PATH)
    print("[OWNER03] using OUT_DIR =", OUT_DIR)

    # owner-side plaintext B (only in local trusted environment)
    B_sp = load_npz(B_PATH).tocsr()
    B_shape = list(B_sp.shape)
    B_nnz = int(B_sp.nnz)
    E = int(B_shape[1])

    # ------------------------------------------------------------------
    # Part A: B_A / B_B / sizeB_A / sizeB_B
    # Default = reuse if already present
    # ------------------------------------------------------------------
    need_build_B = args.rebuild_b or not (
        B_A_path.exists() and B_B_path.exists() and sizeB_A_path.exists() and sizeB_B_path.exists()
    )

    if need_build_B:
        print("[OWNER03] building B shares and sizeB shares ...")

        # plaintext sizeB exists only in memory
        sizeB = np.asarray(B_sp.sum(axis=0)).ravel().astype(np.int64)
        sizeB_A, sizeB_B = share_modp(sizeB, P, rng)
        del sizeB

        # plaintext dense B exists only in memory
        B = B_sp.toarray().astype(np.int64)
        B_A, B_B = share_modp(B, P, rng)
        del B

        np.save(B_A_path, B_A)
        np.save(B_B_path, B_B)
        np.save(sizeB_A_path, sizeB_A)
        np.save(sizeB_B_path, sizeB_B)

        del B_A
        del B_B
        del sizeB_A
        del sizeB_B

        b_mode = "built_now"
        print("[OWNER03] saved:", B_A_path, B_B_path, sizeB_A_path, sizeB_B_path)
    else:
        b_mode = "reused_existing"
        print("[OWNER03] reuse existing B shares and sizeB shares.")

    # ------------------------------------------------------------------
    # Part B: C = B^T B  ->  C_A / C_B
    # This is the new owner-side offline preprocessing for Step2.
    # Plaintext C is NEVER written to disk.
    # ------------------------------------------------------------------
    need_build_C = args.rebuild_c or not (C_A_path.exists() and C_B_path.exists())

    if need_build_C:
        print("[OWNER03] building C shares (owner-side offline preprocessing) ...")

        # plaintext C_sp / C exist only in memory
        C_sp = (B_sp.T @ B_sp).tocsr()
        C_shape = list(C_sp.shape)
        C_nnz = int(C_sp.nnz)

        # keep diagonal as true edge size; do NOT zero it out
        C = C_sp.toarray().astype(np.int64)
        del C_sp

        C_A, C_B = share_modp(C, P, rng)
        del C

        np.save(C_A_path, C_A)
        np.save(C_B_path, C_B)

        del C_A
        del C_B

        c_mode = "built_now"
        print("[OWNER03] saved:", C_A_path, C_B_path)
    else:
        c_mode = "reused_existing"
        print("[OWNER03] reuse existing C shares.")
        # infer shape cheaply
        C_shape = [E, E]
        # if reusing, nnz may be unknown unless meta already has it
        C_nnz = int(meta.get("C_nnz", -1))

    # plaintext sparse B also no longer needed
    del B_sp

    # ------------------------------------------------------------------
    # Update meta
    # ------------------------------------------------------------------
    meta.update({
        "P": int(P),
        "SEED": int(SEED),
        "bpath": str(B_PATH),
        "B_shape": B_shape,
        "B_nnz": B_nnz,
        "C_shape": C_shape,
        "C_nnz": int(C_nnz),
        "B_share_mode": b_mode,
        "C_share_mode": c_mode,
        "contains": {
            "B_A": B_A_path.exists(),
            "B_B": B_B_path.exists(),
            "sizeB_A": sizeB_A_path.exists(),
            "sizeB_B": sizeB_B_path.exists(),
            "C_A": C_A_path.exists(),
            "C_B": C_B_path.exists(),
        },
        "security_note": (
            "Plaintext B, plaintext sizeB, and plaintext C=B^T B are processed "
            "owner-side only. Plaintext sizeB and plaintext C are never written "
            "to disk. Only additive shares are persisted."
        )
    })

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[OWNER03] meta updated:", meta_path)
    print("[OWNER03] done.")

if __name__ == "__main__":
    main()