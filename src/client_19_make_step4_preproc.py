from pathlib import Path
import argparse
import json
import numpy as np
import hashlib
import secrets

P_DEFAULT = 2305843009213693951
SEED_DEFAULT = 20260219

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True,
                    help=r'query dir, e.g. E:\zyc\project\out\queries\queryset\q0013')
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    return ap.parse_args()

def stable_int32_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)

def rand_nonzero_vec(rng: np.random.Generator, p: int, n: int) -> np.ndarray:
    return rng.integers(1, p, size=(n,), dtype=np.int64)

def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)
    STEP4 = Q_DIR / "online_step4_prf"
    STEP4.mkdir(parents=True, exist_ok=True)

    pair_path = STEP4 / "padded_pairs.json"
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing {pair_path}. Run client_16_make_padded_pairs.py first.")

    obj = json.loads(pair_path.read_text(encoding="utf-8"))
    if obj.get("format") != "multi":
        raise ValueError("padded_pairs.json must be multi format.")

    U = int(obj["U"])
    K = int(obj["K"])
    S = int(obj["S"])
    sol_ids = obj.get("sol_ids", list(range(S)))
    total = S * U * K
    CHUNK = int(args.chunk)
    parts = (total + CHUNK - 1) // CHUNK
    P = int(args.p)

    # 为本次 Step 4 执行生成一个公开随机合并系数。
    # 在本次 Step 4 的全部 token 比较中复用该系数。
    merge_c = secrets.randbelow(P)

    seed_eff = int(args.seed) ^ stable_int32_from_str(str(Q_DIR))
    rng = np.random.default_rng(seed_eff)

    offset = 0
    for part in range(parts):
        n = min(CHUNK, total - offset)

        # global nonzero m
        m = rand_nonzero_vec(rng, P, n).astype(object)
        mA = rng.integers(0, P, size=(n,), dtype=np.int64).astype(object)
        mB = (m - mA) % P

        # Beaver triple: c = a*b mod P
        a = rng.integers(0, P, size=(n,), dtype=np.int64).astype(object)
        b = rng.integers(0, P, size=(n,), dtype=np.int64).astype(object)
        c = (a * b) % P

        aA = rng.integers(0, P, size=(n,), dtype=np.int64).astype(object)
        bA = rng.integers(0, P, size=(n,), dtype=np.int64).astype(object)
        cA = rng.integers(0, P, size=(n,), dtype=np.int64).astype(object)

        aB = (a - aA) % P
        bB = (b - bA) % P
        cB = (c - cA) % P

        np.savez_compressed(
            STEP4 / f"step4_preproc_A_part{part}.npz",
            m=np.array(mA, dtype=np.int64),
            a=np.array(aA, dtype=np.int64),
            b=np.array(bA, dtype=np.int64),
            c=np.array(cA, dtype=np.int64),
        )
        np.savez_compressed(
            STEP4 / f"step4_preproc_B_part{part}.npz",
            m=np.array(mB, dtype=np.int64),
            a=np.array(aB, dtype=np.int64),
            b=np.array(bB, dtype=np.int64),
            c=np.array(cB, dtype=np.int64),
        )

        offset += n
        print(f"[client-step4-preproc] wrote part {part}, len={n}")

    meta = {
        "scheme": "masked_zero_test_share_beaver",
        "P": int(P),
        "merge_coefficient": int(merge_c),
        "merge_coefficient_source": "system_random",
        "U": int(U),
        "K": int(K),
        "S": int(S),
        "sol_ids": sol_ids,
        "total": int(total),
        "chunk": int(CHUNK),
        "parts": int(parts),
        "qdir": str(Q_DIR),
        "seed_eff": int(seed_eff),
    }
    (STEP4 / "step4_preproc_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )
    print("[client-step4-preproc] wrote:", STEP4 / "step4_preproc_meta.json")

if __name__ == "__main__":
    main()