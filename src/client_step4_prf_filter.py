from pathlib import Path
import argparse
import json
import numpy as np

P_DEFAULT = 2305843009213693951

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True,
                    help=r'query dir, e.g. E:\zyc\project\out\queries\queryset\q0013')
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--require_injective", action="store_true",
                    help="enforce injective mapping (no two u map to same v) when selecting a solution")
    return ap.parse_args()

def is_injective(final_candidates):
    seen = set()
    for lst in final_candidates:
        if len(lst) != 1:
            return False
        v = lst[0]
        if v in seen:
            return False
        seen.add(v)
    return True

def load_step3_full(step3_full_path: Path):
    if not step3_full_path.exists():
        raise FileNotFoundError(
            f"Missing {step3_full_path}."
            f"Run client_decrypt_step3_dense.py --save_full first."
        )

    obj = json.loads(step3_full_path.read_text(encoding="utf-8"))
    if obj.get("format") != "multi":
        raise ValueError("client_candidates_full.json must be multi format.")

    if "solutions_full_kept" in obj and isinstance(obj["solutions_full_kept"], list):
        sols = obj["solutions_full_kept"]
    elif "solutions" in obj and isinstance(obj["solutions"], list):
        sols = obj["solutions"]
    else:
        raise ValueError("client_candidates_full.json missing solutions / solutions_full_kept")

    sol_map = {}
    for s in sols:
        sid = s.get("sol_id", None)
        u_cands = s.get("u_candidates", None)
        if isinstance(sid, int) and isinstance(u_cands, list):
            sol_map[int(sid)] = u_cands

    return obj, sol_map

def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)
    STEP4 = Q_DIR / "online_step4_prf"
    STEP3 = Q_DIR / "online_step3"

    pair_path = STEP4 / "padded_pairs.json"
    meta_path = STEP4 / "cloud_meta.json"
    out_path = STEP4 / "final_after_prf.json"
    step3_full_path = STEP3 / "client_candidates_full.json"

    if not pair_path.exists():
        raise FileNotFoundError(f"Missing {pair_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run cloudA_step4_prf_eq_onlypairs.py first.")

    pairs = json.loads(pair_path.read_text(encoding="utf-8"))
    if pairs.get("format") != "multi":
        raise ValueError("padded_pairs.json must be multi format with padded_pairs[S][U][K].")

    padded_all = pairs["padded_pairs"]
    U = int(pairs["U"])
    K = int(pairs["K"])
    S = int(pairs["S"])
    sol_ids = pairs.get("sol_ids", list(range(S)))

    _, step3_sol_map = load_step3_full(step3_full_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    P = int(meta.get("P", args.p))
    parts = int(meta["parts"])
    total = int(meta["total"])
    c = int(meta["c"])

    ok_flat = np.zeros((total,), dtype=np.uint8)

    offset = 0
    for part in range(parts):
        a_part = STEP4 / f"cloudA_z_part{part}.npz"
        b_part = STEP4 / f"cloudB_z_part{part}.npz"
        if not a_part.exists() or not b_part.exists():
            raise FileNotFoundError(
                f"Missing parts: {a_part} / {b_part}. "
                f"Run cloudA_step4_finalize.py and cloudB_step4_prf_eq_onlypairs.py first."
            )

        A = np.load(a_part)["z"].astype(np.int64) % P
        B = np.load(b_part)["z"].astype(np.int64) % P
        if A.shape != B.shape:
            raise ValueError(f"Bad shape at part {part}: A={A.shape}, B={B.shape}")

        z = (A.astype(object) + B.astype(object)) % P
        eq = (np.array(z, dtype=object) == 0)
        n = eq.shape[0]
        ok_flat[offset:offset+n] = eq.astype(np.uint8)
        offset += n

    if offset != total:
        raise ValueError(f"offset mismatch: {offset} vs total {total}")

    ok = ok_flat.reshape((S, U, K))

    finals = []
    stats = []
    good_solution_indices = []

    for s in range(S):
        padded = padded_all[s]
        sol_id = int(sol_ids[s]) if s < len(sol_ids) else int(s)

        if sol_id not in step3_sol_map:
            raise ValueError(
                f"sol_id={sol_id} from padded_pairs.json not found in {step3_full_path}"
            )

        true_u_candidates = step3_sol_map[sol_id]
        if len(true_u_candidates) != U:
            raise ValueError(
                f"sol_id={sol_id}: len(true_u_candidates)={len(true_u_candidates)} != U={U}"
            )

        final = []
        sizes = []

        for u in range(U):
            row = padded[u]
            true_set = set(int(x) for x in true_u_candidates[u])

            keep = [int(row[j]) for j in range(K)
                    if ok[s, u, j] == 1 and int(row[j]) in true_set]

            final.append(keep)
            sizes.append(len(keep))

        num_zero = int(sum(1 for x in sizes if x == 0))
        num_unique = int(sum(1 for x in sizes if x == 1))
        feasible = (num_zero == 0)

        inj_ok = True
        if feasible and args.require_injective:
            inj_ok = is_injective(final)

        if feasible and inj_ok:
            good_solution_indices.append(s)

        stat = {
            "sol_id": sol_id,
            "idx_in_padded": int(s),
            "feasible_no_zero": bool(feasible),
            "injective_ok": bool(inj_ok),
            "after_min_avg_max": [int(min(sizes)), float(np.mean(sizes)), int(max(sizes))],
            "num_unique": int(num_unique),
            "num_zero": int(num_zero),
            "preview_first10": {f"u{i}": final[i][:20] for i in range(min(10, U))}
        }
        finals.append(final)
        stats.append(stat)

    chosen = None
    chosen_idx = None
    if len(good_solution_indices) == 1:
        chosen_idx = good_solution_indices[0]
        chosen = finals[chosen_idx]
    elif len(good_solution_indices) > 1:
        best = None
        for idx in good_solution_indices:
            st = stats[idx]
            score = (st["num_unique"], -st["after_min_avg_max"][1])
            if best is None or score > best[0]:
                best = (score, idx)
        chosen_idx = best[1]
        chosen = finals[chosen_idx]

    out = {
        "scheme": "masked_zero_test_share_beaver",
        "merge_limbs": True,
        "c": int(c),
        "qdir": str(Q_DIR),
        "U": U, "K": K, "S": S,
        "sol_ids": sol_ids,
        "require_injective": bool(args.require_injective),
        "good_solution_indices": good_solution_indices,
        "chosen_solution_index": chosen_idx,
        "chosen_sol_id": (int(sol_ids[chosen_idx]) if chosen_idx is not None and chosen_idx < len(sol_ids) else None),
        "all_solutions_stats": stats,
        "chosen_final_candidates": chosen,
    }

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_path)
    print("good solutions =", len(good_solution_indices),
          "chosen_idx =", chosen_idx,
          "chosen_sol_id =", out.get("chosen_sol_id"))

if __name__ == "__main__":
    main()