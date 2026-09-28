from pathlib import Path
import argparse
import json
import numpy as np

P_DEFAULT = 2305843009213693951
SEED_DEFAULT = 20260128

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner_prf_dir", type=str, default=r"E:\zyc\project\out\hc_owner\prf",
                    help="dir containing data_tok_plain.npy")
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    return ap.parse_args()

def share_modp_uint64_safe(X, p: int, rng: np.random.Generator):
    """
    X may be uint64 (values up to 2^64-1). We must NOT cast to int64 before mod.
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
    owner = Path(args.owner_prf_dir)
    owner.mkdir(parents=True, exist_ok=True)

    plain_path = owner / "data_tok_plain.npy"
    if not plain_path.exists():
        raise FileNotFoundError(f"Missing {plain_path}. Run owner_01_make_data_prf_tokens.py first.")

    P = int(args.p)
    rng = np.random.default_rng(int(args.seed))

    tok = np.load(plain_path)  # should be (V,2) uint64
    if tok.dtype != np.uint64:
        tok = tok.astype(np.uint64)

    A, B = share_modp_uint64_safe(tok, P, rng)

    np.save(owner / "data_tok_A.npy", A)
    np.save(owner / "data_tok_B.npy", B)

    (owner / "share_params.json").write_text(
        json.dumps({
            "P": int(P),
            "SEED": int(args.seed),
            "owner_prf_dir": str(owner)
        }, indent=2),
        encoding="utf-8"
    )
    print("Saved:", owner / "data_tok_A.npy", owner / "data_tok_B.npy")
    print("Meta:", owner / "share_params.json")

if __name__ == "__main__":
    main()