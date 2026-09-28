from pathlib import Path
import argparse, json, hashlib
import numpy as np

P_DEFAULT = 2305843009213693951
BASE_SEED_DEFAULT = 20260201

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, default=r"E:\zyc\project\out\queries\q1",
                    help="single query dir, e.g. ...\\queryset\\q0008")
    ap.add_argument("--p", type=int, default=P_DEFAULT, help="prime modulus")
    ap.add_argument("--base_seed", type=int, default=BASE_SEED_DEFAULT,
                    help="base seed used to derive per-query seed (stable across runs)")
    return ap.parse_args()

def derive_seed(base_seed: int, name: str) -> int:
    # stable 32-bit seed derived from (base_seed + foldername)
    s = f"{base_seed}:{name}".encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:4], "little", signed=False)

def gen_alpha_for_qdir(qdir: Path, p: int, base_seed: int):
    """
    Generate alpha for THIS qdir only.
    Writes: qdir/online_step3/alpha.json
    """
    S = qdir / "shares_dense"
    QA_path = S / "Q_A.npy"
    if not QA_path.exists():
        raise FileNotFoundError(f"Missing {QA_path}. Run 06_make_dense_shares.py --qdir first.")

    Q_A = np.load(QA_path)
    Eq = int(Q_A.shape[1])

    seed = derive_seed(base_seed, qdir.name)
    rng = np.random.default_rng(seed)

    # alpha in [1, p-1], length Eq
    alpha = rng.integers(1, p, size=Eq, dtype=np.int64).tolist()

    OUT = qdir / "online_step3"
    OUT.mkdir(parents=True, exist_ok=True)

    payload = {
        "P": int(p),
        "alpha": alpha,
        "seed": int(seed),
        "Eq": int(Eq),
        "base_seed": int(base_seed),
        "qdir": str(qdir),
    }
    (OUT / "alpha.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[client-make-alpha] wrote: {OUT/'alpha.json'} (Eq={Eq}, seed={seed})")

def main():
    args = parse_args()
    qdir = Path(args.qdir)
    p = int(args.p)
    base_seed = int(args.base_seed)

    gen_alpha_for_qdir(qdir, p, base_seed)

if __name__ == "__main__":
    main()
