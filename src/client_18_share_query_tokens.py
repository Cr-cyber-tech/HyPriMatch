from pathlib import Path
import numpy as np
import json
import argparse

P_DEFAULT = 2305843009213693951
SEED_DEFAULT = 20260128

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True,
                    help=r'query dir, e.g. E:\zyc\project\out\queries\queryset\q0013')
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    return ap.parse_args()

def share_modp_uint64_safe(X, p: int, rng: np.random.Generator):
    """
    X may be uint64; do NOT cast to int64 before mod.
    Return int64 shares in [0, p).
    """
    Xo = X.astype(object)
    Xmod = (Xo % p)

    A = rng.integers(0, p, size=X.shape, dtype=np.int64)
    Ao = A.astype(object)
    Bo = (Xmod - Ao) % p
    B = Bo.astype(np.int64)
    return A, B

def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)
    STEP4 = Q_DIR / "online_step4_prf"
    STEP4.mkdir(parents=True, exist_ok=True)

    q_path = STEP4 / "query_tok_plain.npy"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing {q_path}. Run client_17_make_query_prf_tokens.py --qdir first.")

    P = int(args.p)
    rng = np.random.default_rng(int(args.seed))

    q = np.load(q_path)  # (U,2) uint64
    if q.dtype != np.uint64:
        q = q.astype(np.uint64)

    qA, qB = share_modp_uint64_safe(q, P, rng)
    np.save(STEP4 / "query_tok_A.npy", qA)
    np.save(STEP4 / "query_tok_B.npy", qB)

    (STEP4 / "query_share_params.json").write_text(
        json.dumps({"P": int(P), "SEED": int(args.seed), "qdir": str(Q_DIR)}, indent=2),
        encoding="utf-8"
    )
    print("Saved:", STEP4 / "query_tok_A.npy", STEP4 / "query_tok_B.npy")

if __name__ == "__main__":
    main()