from pathlib import Path
import json
import numpy as np
import argparse

CHUNK_V_DEFAULT = 256

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True)
    ap.add_argument("--chunk_v", type=int, default=CHUNK_V_DEFAULT)
    ap.add_argument("--global_data_share_dir", type=str, default=None,
                    help="global dir containing B_B.npy")
    return ap.parse_args()

def load_alpha(step3_dir: Path):
    a = json.loads((step3_dir / "alpha.json").read_text(encoding="utf-8"))
    alpha = np.array(a["alpha"], dtype=np.int64)
    p = int(a["P"])
    return alpha, p

def load_solutions(step12_dir: Path):
    e = json.loads((step12_dir / "matched_edges.json").read_text(encoding="utf-8"))
    if e.get("format") != "multi":
        raise ValueError("matched_edges.json is not multi format. Expected e['format']=='multi'.")
    sols = e.get("matched_edges_solutions", None)
    if not isinstance(sols, list) or len(sols) == 0:
        raise ValueError("matched_edges_solutions missing/empty in matched_edges.json")
    return sols

def proj_mod_bigint(M_int64, alpha_int64, p):
    M = M_int64.astype(object)
    alpha = alpha_int64.astype(object)
    out = np.zeros((M.shape[0],), dtype=object)
    for j in range(M.shape[1]):
        out = (out + M[:, j] * alpha[j]) % p
    return out.astype(np.int64)

def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)
    CHUNK_V = int(args.chunk_v)

    S = Q_DIR / "shares_dense"
    global_data_share_dir = Path(args.global_data_share_dir) if args.global_data_share_dir else None
    if global_data_share_dir is None:
        raise ValueError("--global_data_share_dir is required")
    
    STEP3_DIR = Q_DIR / "online_step3"
    STEP3_DIR.mkdir(parents=True, exist_ok=True)
    STEP12_DIR = Q_DIR / "online_step12_cloud"

    print("[DEBUG] using Q_DIR =", Q_DIR)
    print("[DEBUG] shares =", S)
    print("[DEBUG] out    =", STEP3_DIR)

    alpha, p = load_alpha(STEP3_DIR)
    solutions = load_solutions(STEP12_DIR)

    B_B = np.load(global_data_share_dir / "B_B.npy")
    Q_B = np.load(S / "Q_B.npy")

    V, E = B_B.shape
    U, Eq = Q_B.shape

    qproj = proj_mod_bigint(Q_B, alpha, p)  # (U,)

    num_parts = (V + CHUNK_V - 1) // CHUNK_V

    for si, matched_edges in enumerate(solutions):
        if len(matched_edges) != Eq:
            raise ValueError(f"solution {si} length {len(matched_edges)} != Eq {Eq}")

        sol_dir = STEP3_DIR / f"sol{si:04d}"
        sol_dir.mkdir(parents=True, exist_ok=True)

        Bsub = B_B[:, matched_edges]  # (V,Eq)

        for part in range(num_parts):
            v0 = part * CHUNK_V
            v1 = min(V, v0 + CHUNK_V)

            bproj = proj_mod_bigint(Bsub[v0:v1, :], alpha, p)
            rB = (qproj[:, None] - bproj[None, :]) % p

            np.savez_compressed(sol_dir / f"cloudB_r_part{part}.npz", r=rB.astype(np.int64))

        print(f"[cloudB-step3] sol{si:04d} done. matched_edges={matched_edges}")

if __name__ == "__main__":
    main()
