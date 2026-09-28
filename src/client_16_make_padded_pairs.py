from pathlib import Path
import json
import numpy as np
import math
import argparse
import hashlib

SEED_DEFAULT = 20260125
PAD_RATIO_DEFAULT = 1.10

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True,
                    help=r'query dir, e.g. E:\zyc\project\out\queries\queryset\q0013')
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--pad_ratio", type=float, default=PAD_RATIO_DEFAULT)
    ap.add_argument("--max_solutions", type=int, default=0,
                    help="0 means use all kept solutions; otherwise only take first N kept solutions")
    return ap.parse_args()

def derive_seed(base_seed: int, sid: int) -> int:
    # stable 32-bit seed derived from (base_seed, sid)
    s = f"{base_seed}:{sid}".encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:4], "little", signed=False)

def load_full_solutions(full_obj: dict):
    """
    兼容多种 full 文件 schema：
      - full["solutions"]
      - full["solutions_full_kept"]
      - full["solutions_full_all"]
    """
    for key in ["solutions", "solutions_full_kept", "solutions_full_all"]:
        v = full_obj.get(key, None)
        if isinstance(v, list) and len(v) > 0:
            return v, key
    return None, None

def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)

    STEP3 = Q_DIR / "online_step3"
    IN_SUM = STEP3 / "client_candidates.json"
    IN_FULL = STEP3 / "client_candidates_full.json"

    OUT_DIR = Q_DIR / "online_step4_prf"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_SUM.exists():
        raise FileNotFoundError(f"Missing {IN_SUM}. Run step3 client decrypt first.")
    if not IN_FULL.exists():
        raise FileNotFoundError(f"Missing {IN_FULL}. Re-run step3 with --save_full so full candidates are saved.")

    summ = json.loads(IN_SUM.read_text(encoding="utf-8"))
    if summ.get("format") != "multi":
        raise ValueError("client_candidates.json is not multi format.")

    kept = summ.get("kept_solution_ids_no_zero", None)
    if not isinstance(kept, list) or len(kept) == 0:
        raise ValueError("kept_solution_ids_no_zero missing/empty in client_candidates.json")

    full = json.loads(IN_FULL.read_text(encoding="utf-8"))
    if full.get("format") != "multi":
        raise ValueError(
            "client_candidates_full.json is not multi format OR you saved a stub. "
            "Please run step3 with --save_full to include u_candidates."
        )

    # U/V 以 full 为准（full 没有就 fallback 到 summ）
    U = int(full.get("U", summ.get("U")))
    V = int(full.get("V", summ.get("V")))

    full_solutions, used_key = load_full_solutions(full)
    if not isinstance(full_solutions, list) or len(full_solutions) == 0:
        raise ValueError(
            "client_candidates_full.json missing solutions with u_candidates.\n"
            "I tried keys: solutions / solutions_full_kept / solutions_full_all.\n"
            "Re-run step3 with --save_full OR check your step3 writer."
        )

    # sol_id -> u_candidates
    sol_map = {}
    for s in full_solutions:
        sid = s.get("sol_id", None)
        u_cands = s.get("u_candidates", None)
        if isinstance(sid, int) and isinstance(u_cands, list):
            sol_map[int(sid)] = u_cands

    if len(sol_map) == 0:
        raise ValueError(
            f"Found full solutions under '{used_key}', but none has (sol_id, u_candidates). "
            "Your full file schema may have changed again."
        )

    kept = [int(x) for x in kept if int(x) in sol_map]
    if int(args.max_solutions) > 0:
        kept = kept[:int(args.max_solutions)]
    if len(kept) == 0:
        raise ValueError(
            "After intersecting kept_solution_ids_no_zero with full solutions, kept is empty.\n"
            "This usually means your full file doesn't contain the kept solution ids."
        )

    # ===== 计算 K_global：同一个 query 的所有 kept solutions 统一 K =====
    max_c = 0
    for sid in kept:
        u_cands = sol_map[sid]
        if len(u_cands) != U:
            raise ValueError(f"solution {sid}: len(u_candidates)={len(u_cands)} != U={U}")
        for u in range(U):
            max_c = max(max_c, len(u_cands[u]))

    PAD_RATIO = float(args.pad_ratio)
    K = int(math.ceil(max_c * PAD_RATIO))
    if K < max_c:
        K = max_c

    # padded_pairs: [S][U][K]
    padded_all = []
    for sid in kept:
        u_cands = sol_map[sid]

        # ✅ 关键改动：seed 绑定 sid，避免不同 solution 的 fake 完全同步
        sid_seed = derive_seed(int(args.seed), int(sid))
        rng = np.random.default_rng(sid_seed)

        padded = []
        for u in range(U):
            true = list(u_cands[u])
            true_set = set(true)

            need = K - len(true)
            fake = []
            # 注意：避免死循环（V 很大一般不会卡住）
            tries = 0
            while len(fake) < need:
                v = int(rng.integers(0, V))
                tries += 1
                if v in true_set:
                    continue
                fake.append(v)
                if tries > 10_000_000:
                    raise RuntimeError("Padding seems stuck; V too small or candidate set too large?")

            mixed = true + fake
            rng.shuffle(mixed)
            padded.append(mixed)

        padded_all.append(padded)

    out = {
        "format": "multi",
        "qdir": str(Q_DIR),
        "U": int(U),
        "V": int(V),
        "S": int(len(kept)),
        "sol_ids": kept,
        "K": int(K),
        "seed": int(args.seed),
        "pad_ratio": PAD_RATIO,
        "max_true_candidates_over_kept": int(max_c),
        "full_source_key": used_key,
        "padded_pairs": padded_all  # shape [S][U][K]
    }
    (OUT_DIR / "padded_pairs.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    meta = {
        "qdir": str(Q_DIR),
        "U": int(U), "V": int(V),
        "S": int(len(kept)),
        "sol_ids": kept,
        "K": int(K),
        "max_true_candidates_over_kept": int(max_c),
        "pad_ratio": PAD_RATIO,
        "seed_base": int(args.seed),
        "seed_rule": "seed_per_solution = sha256(f'{seed_base}:{sol_id}')[:4] little-endian",
        "from": {
            "kept_solution_ids_no_zero": str(IN_SUM),
            "u_candidates_full": str(IN_FULL),
            "full_source_key": used_key
        }
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", OUT_DIR / "padded_pairs.json")
    print("K_global =", K, "max_true_candidates_over_kept =", max_c, "S_kept =", len(kept), "full_key =", used_key)

if __name__ == "__main__":
    main()