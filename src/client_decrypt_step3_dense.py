from pathlib import Path
import json
import numpy as np
import argparse

P_DEFAULT = 2305843009213693951
CHUNK_DEFAULT = 256

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True,
                    help="query dir, e.g. ...\\out\\queries\\queryset\\q0008")
    ap.add_argument("--chunk_v", type=int, default=CHUNK_DEFAULT,
                    help="V chunk size (must match cloud step3 chunk)")
    ap.add_argument("--p", type=int, default=None,
                    help="prime modulus override (optional)")
    ap.add_argument("--global_data_share_dir", type=str, default=None,
                    help="global dir containing shared data-hypergraph tensors or metadata")

    # ✅ 新增：默认保存 kept 的 full；如需保存所有解的 full，用这个开关
    ap.add_argument("--save_full_all", action="store_true",
                    help="also save full u_candidates for ALL solutions (can be huge)")
    return ap.parse_args()

def load_alpha_p(step3_dir: Path):
    a_path = step3_dir / "alpha.json"
    if not a_path.exists():
        raise FileNotFoundError(f"Missing {a_path}. Run client_make_alpha first.")
    a = json.loads(a_path.read_text(encoding="utf-8"))
    return int(a["P"])

def load_solutions(step12_dir: Path):
    e_path = step12_dir / "matched_edges.json"
    if not e_path.exists():
        raise FileNotFoundError(f"Missing {e_path}. Run step12 driver first.")
    e = json.loads(e_path.read_text(encoding="utf-8"))
    if e.get("format") != "multi":
        raise ValueError("matched_edges.json is not multi format. Expected e['format']=='multi'.")
    sols = e.get("matched_edges_solutions", None)
    if not isinstance(sols, list) or len(sols) == 0:
        raise ValueError("matched_edges_solutions missing/empty in matched_edges.json")
    return sols, e

def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)

    S = Q_DIR / "shares_dense"
    STEP3_DIR = Q_DIR / "online_step3"
    STEP12_DIR = Q_DIR / "online_step12_cloud"
    STEP3_DIR.mkdir(parents=True, exist_ok=True)

    CHUNK_V = int(args.chunk_v)

    p_from_alpha = load_alpha_p(STEP3_DIR)
    P = int(args.p) if args.p is not None else int(p_from_alpha)

    # shapes from shares_dense
    QA_path = S / "Q_A.npy"
    if not QA_path.exists():
        raise FileNotFoundError(f"Missing {QA_path}. Run 06_make_dense_shares.py --qdir first.")

    Q_A = np.load(QA_path)
    U, Eq = Q_A.shape

    # ===== 数据超图维度 V：优先从全局数据 share / metadata 读取 =====
    if args.global_data_share_dir is not None:
        G = Path(args.global_data_share_dir)
        meta_path = G / "data_share_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            B_shape = meta.get("B_shape", None)
            if not isinstance(B_shape, list) or len(B_shape) != 2:
                raise ValueError(f"Invalid B_shape in {meta_path}: {B_shape}")
            V = int(B_shape[0])
        else:
            BA_global = G / "B_A.npy"
            if not BA_global.exists():
                raise FileNotFoundError(
                    f"Missing both {meta_path} and {BA_global}. "
                    f"Need global data share metadata or B_A.npy."
                )
            B_A = np.load(BA_global, mmap_mode="r")
            V = int(B_A.shape[0])
    else:
        BA_path = S / "B_A.npy"
        if not BA_path.exists():
            raise FileNotFoundError(
                "Missing {BA_path}. "
                f"If using global data shares, rerun with --global_data_share_dir."
            )
        B_A = np.load(BA_path, mmap_mode="r")
        V = int(B_A.shape[0])

    solutions, meta_step12 = load_solutions(STEP12_DIR)
    num_parts = (V + CHUNK_V - 1) // CHUNK_V

    kept_solution_ids = []
    solutions_items = []

    # ✅ full：默认保存 kept 的 full；可选保存全部
    full_kept = []
    full_all = [] if args.save_full_all else None

    for si, matched_edges in enumerate(solutions):
        sol_dir = STEP3_DIR / f"sol{si:04d}"
        if not sol_dir.exists():
            raise FileNotFoundError(
                f"Missing {sol_dir}. Run cloudA_step3_dense.py / cloudB_step3_dense.py for this qdir first."
            )

        u_candidates = [[] for _ in range(U)]

        for part in range(num_parts):
            a_part = sol_dir / f"cloudA_r_part{part}.npz"
            b_part = sol_dir / f"cloudB_r_part{part}.npz"
            if not a_part.exists() or not b_part.exists():
                raise FileNotFoundError(f"Missing step3 parts for sol={si} part={part}: {a_part} / {b_part}")

            A = np.load(a_part)["r"]
            B = np.load(b_part)["r"]
            r = (A + B) % P

            v_start = part * CHUNK_V
            us, vs = np.where(r == 0)
            for uu, vv in zip(us.tolist(), vs.tolist()):
                u_candidates[uu].append(v_start + vv)

        sizes = [len(x) for x in u_candidates]
        num_zero = int(sum(1 for s in sizes if s == 0))
        num_unique = int(sum(1 for s in sizes if s == 1))
        feasible = (num_zero == 0)

        if feasible:
            kept_solution_ids.append(si)
            full_kept.append({
                "sol_id": int(si),
                "matched_edges_in_data": matched_edges,
                "u_candidates": u_candidates
            })

        if args.save_full_all:
            full_all.append({
                "sol_id": int(si),
                "matched_edges_in_data": matched_edges,
                "feasible_no_zero": bool(feasible),
                "u_candidates": u_candidates
            })

        item = {
            "sol_id": int(si),
            "matched_edges_in_data": matched_edges,
            "feasible_no_zero": bool(feasible),
            "candidate_sizes_min_avg_max": [int(min(sizes)), float(np.mean(sizes)), int(max(sizes))],
            "num_zero": num_zero,
            "num_unique": num_unique,
            "unique_rate": float(num_unique / U),
            "preview_first10": {f"u{i}": u_candidates[i][:20] for i in range(min(10, U))}
        }
        solutions_items.append(item)

        print(f"[client-step3] sol{si:04d} done. feasible={feasible}, num_zero={num_zero}, num_unique={num_unique}")

    # ✅ 写简表（保持你原来的文件名）
    out_path = STEP3_DIR / "client_candidates.json"
    out = {
        "format": "multi",
        "qdir": str(Q_DIR),
        "U": int(U),
        "V": int(V),
        "Eq": int(Eq),
        "p": int(P),
        "chunk_v": int(CHUNK_V),
        "step12_meta": {
            "num_solutions": int(meta_step12.get("num_solutions", len(solutions))),
            "max_solutions": meta_step12.get("max_solutions", None),
            "allow_reuse": meta_step12.get("allow_reuse", None),
            "triple_seed": meta_step12.get("triple_seed", None),
        },
        "kept_solution_ids_no_zero": kept_solution_ids,
        "n_solutions_total": int(len(solutions)),
        "n_solutions_kept": int(len(kept_solution_ids)),
        "solutions": solutions_items
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_path)

    # ✅ 写 full（默认：至少写 kept 的 full，保证 Step4 可用）
    full_path = STEP3_DIR / "client_candidates_full.json"
    full_dump = {
        "format": "multi",
        "qdir": str(Q_DIR),
        "U": int(U),
        "V": int(V),
        "Eq": int(Eq),
        "chunk_v": int(CHUNK_V),
        "p": int(P),
        "kept_solution_ids_no_zero": kept_solution_ids,
        "solutions_full_kept": full_kept,
        "note": "contains full u_candidates for kept (feasible_no_zero) solutions only"
    }

    if args.save_full_all:
        full_dump["solutions_full_all"] = full_all
        full_dump["note"] = "contains full u_candidates for ALL solutions (may be huge)"

    full_path.write_text(json.dumps(full_dump, ensure_ascii=False), encoding="utf-8")
    print("Also saved:", full_path)

if __name__ == "__main__":
    main()