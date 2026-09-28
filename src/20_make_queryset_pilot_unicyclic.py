from pathlib import Path
import json
import math
import argparse
import numpy as np
from scipy.sparse import load_npz, save_npz


B_PATH_DEFAULT = r"E:\zyc\project\out_ma\ma_incidence_VxE.npz"
OUT_ROOT_DEFAULT = r"E:\zyc\project\out_ma\queries\pilot_structural_protocol_eq5"


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--bpath", type=str, default=B_PATH_DEFAULT,
                    help="path to incidence matrix npz")
    ap.add_argument("--out_root", type=str, default=OUT_ROOT_DEFAULT,
                    help="output root dir")
    ap.add_argument("--n_queries", type=int, default=10,
                    help="number of queries to generate")
    ap.add_argument("--seed", type=int, default=20260201)

    # target edge count for sweep-pilot
    ap.add_argument("--eq_target", type=int, required=True,
                    help="target number of hyperedges in each query")

    # vertex range
    ap.add_argument("--u_min", type=int, default=0)
    ap.add_argument("--u_max", type=int, default=140)

    # optional manual overrides; if omitted, derived from eq_target
    ap.add_argument("--tau", type=int, default=None)
    ap.add_argument("--min_num_overlap_pairs", type=int, default=None)
    ap.add_argument("--min_distinct_intersections", type=int, default=None)
    ap.add_argument("--min_distinct_sizes", type=int, default=None)
    ap.add_argument("--strong_overlap_tau", type=int, default=None)
    ap.add_argument("--min_num_strong_overlap_pairs", type=int, default=None)

    # structure flags
    ap.add_argument("--require_connected_overlap_graph", action="store_true", default=True)
    ap.add_argument("--require_all_pairs_overlap", action="store_true", default=False)

    # sampling control
    ap.add_argument("--max_tries", type=int, default=200000)
    ap.add_argument("--progress_every", type=int, default=5000)

    return ap.parse_args()


def edge_vertices_from_B(B):
    Bc = B.tocsc()
    E = Bc.shape[1]
    verts = []
    for e in range(E):
        v_idx = Bc.indices[Bc.indptr[e]:Bc.indptr[e + 1]]
        verts.append(v_idx.astype(np.int64))
    return verts


def is_connected_from_adj(adj):
    n = len(adj)
    if n == 0:
        return False
    vis = [False] * n
    stack = [0]
    vis[0] = True
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not vis[y]:
                vis[y] = True
                stack.append(y)
    return all(vis)


def build_neighbor_sets_from_C(C, threshold):
    """
    Build neighbor sets under absolute intersection threshold.
    """
    E = C.shape[0]
    neigh_sets = []
    deg = np.zeros(E, dtype=np.int64)

    for e in range(E):
        row = C.getrow(e)
        js = row.indices
        vals = row.data

        good = []
        for j, v in zip(js.tolist(), vals.tolist()):
            if int(v) >= threshold:
                good.append(int(j))

        s = set(good)
        neigh_sets.append(s)
        deg[e] = len(s)

    return neigh_sets, deg


def induced_submatrix_from_sparse(C, chosen):
    """
    Return dense pairwise intersection matrix for the selected edges.
    chosen: list[int]
    """
    sub = C[chosen, :][:, chosen].toarray().astype(np.int64)
    np.fill_diagonal(sub, 0)
    return sub


def build_overlap_graph_from_inter(inter, tau):
    Eq = inter.shape[0]
    adj = [[] for _ in range(Eq)]
    overlap_pairs = []

    for i in range(Eq):
        for j in range(i + 1, Eq):
            if int(inter[i, j]) >= tau:
                adj[i].append(j)
                adj[j].append(i)
                overlap_pairs.append((i, j, int(inter[i, j])))

    return adj, overlap_pairs


def full_check(
    chosen,
    chosen_nodes,
    edge_sizes,
    C,
    tau,
    min_num_overlap_pairs,
    require_connected_overlap_graph,
    require_all_pairs_overlap,
    min_distinct_intersections,
    min_distinct_sizes,
    strong_overlap_tau,
    min_num_strong_overlap_pairs,
    u_min,
    u_max,
):
    Eq = len(chosen)
    U = len(chosen_nodes)

    if not (u_min <= U <= u_max):
        return False, "U_out_of_range", None

    inter = induced_submatrix_from_sparse(C, chosen)
    adj, overlap_pairs = build_overlap_graph_from_inter(inter, tau)

    num_overlap_pairs = len(overlap_pairs)
    total_pairs = Eq * (Eq - 1) // 2

    if require_all_pairs_overlap and num_overlap_pairs != total_pairs:
        return False, "not_all_pairs_overlap", None

    if num_overlap_pairs < min_num_overlap_pairs:
        return False, "overlap_pairs_too_few", None

    if require_connected_overlap_graph and not is_connected_from_adj(adj):
        return False, "not_connected", None

    overlap_degrees = [len(x) for x in adj]

    distinct_intersections = sorted(set(v for _, _, v in overlap_pairs))
    if len(distinct_intersections) < min_distinct_intersections:
        return False, "distinct_intersections_too_few", None

    chosen_sizes = [int(edge_sizes[e]) for e in chosen]
    distinct_sizes = sorted(set(chosen_sizes))
    if len(distinct_sizes) < min_distinct_sizes:
        return False, "distinct_sizes_too_few", None

    strong_pairs = 0
    for _, _, v in overlap_pairs:
        if int(v) >= strong_overlap_tau:
            strong_pairs += 1

    if strong_pairs < min_num_strong_overlap_pairs:
        return False, "strong_overlap_pairs_too_few", None

    meta = {
        "Eq": int(Eq),
        "U": int(U),
        "num_overlap_pairs": int(num_overlap_pairs),
        "overlap_degrees": overlap_degrees,
        "distinct_intersections": distinct_intersections,
        "distinct_sizes": distinct_sizes,
        "strong_pairs": int(strong_pairs),
        "chosen_sizes": chosen_sizes,
        "intersections_matrix": inter.tolist(),
    }
    return True, "ok", meta


def sample_one_query(
    rng,
    tau_neigh_sets,
    strong_neigh_sets,
    tau_degrees,
    edge_vertices,
    edge_sizes,
    C,
    eq_target,
    tau,
    min_num_overlap_pairs,
    require_connected_overlap_graph,
    require_all_pairs_overlap,
    min_distinct_intersections,
    min_distinct_sizes,
    strong_overlap_tau,
    min_num_strong_overlap_pairs,
    u_min,
    u_max,
):
    E = len(tau_neigh_sets)

    # ---------- Step 1: choose center edge ----------
    weights = tau_degrees.astype(np.float64) + 1e-6
    if weights.sum() <= 0:
        center = int(rng.integers(0, E))
    else:
        weights /= weights.sum()
        center = int(rng.choice(np.arange(E), p=weights))

    chosen = [center]
    chosen_set = {center}
    chosen_nodes = set(int(x) for x in edge_vertices[center])

    # ---------- Step 2: overlap-guided expansion ----------
    while len(chosen) < eq_target:
        frontier = set()
        for e in chosen:
            frontier.update(tau_neigh_sets[e])
        frontier -= chosen_set

        if not frontier:
            return None, "frontier_empty"

        frontier_list = []
        frontier_weights = []

        for cand in frontier:
            cand = int(cand)
            cand_nodes = set(int(x) for x in edge_vertices[cand])
            u_new = len(chosen_nodes | cand_nodes)
            if u_new > u_max:
                continue

            # local guidance score:
            # prefer edges overlapping more with current chosen set,
            # and slightly prefer stronger overlaps
            tau_hits = 0
            strong_hits = 0
            for e in chosen:
                if cand in tau_neigh_sets[e]:
                    tau_hits += 1
                if cand in strong_neigh_sets[e]:
                    strong_hits += 1

            if tau_hits == 0:
                continue

            score = float(tau_hits) + 0.5 * float(strong_hits)
            frontier_list.append(cand)
            frontier_weights.append(score)

        if not frontier_list:
            return None, "frontier_all_rejected_by_u"

        fw = np.array(frontier_weights, dtype=np.float64)
        fw /= fw.sum()

        picked = int(rng.choice(np.array(frontier_list, dtype=np.int64), p=fw))
        chosen.append(picked)
        chosen_set.add(picked)
        chosen_nodes |= set(int(x) for x in edge_vertices[picked])

    # ---------- Step 3: full check ----------
    ok, reason, meta = full_check(
        chosen=chosen,
        chosen_nodes=chosen_nodes,
        edge_sizes=edge_sizes,
        C=C,
        tau=tau,
        min_num_overlap_pairs=min_num_overlap_pairs,
        require_connected_overlap_graph=require_connected_overlap_graph,
        require_all_pairs_overlap=require_all_pairs_overlap,
        min_distinct_intersections=min_distinct_intersections,
        min_distinct_sizes=min_distinct_sizes,
        strong_overlap_tau=strong_overlap_tau,
        min_num_strong_overlap_pairs=min_num_strong_overlap_pairs,
        u_min=u_min,
        u_max=u_max,
    )
    if not ok:
        return None, reason

    return {
        "chosen_edges": chosen,
        "chosen_nodes": sorted(chosen_nodes),
        "meta": meta,
    }, "ok"


def main():
    args = parse_args()

    B_PATH = Path(args.bpath)
    OUT_ROOT = Path(args.out_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    N_QUERIES = int(args.n_queries)
    SEED = int(args.seed)
    EQ = int(args.eq_target)
    U_MIN = int(args.u_min)
    U_MAX = int(args.u_max)
    MAX_TRIES = int(args.max_tries)
    PROGRESS_EVERY = int(args.progress_every)

    # derived defaults
    # derived defaults
    total_pairs = EQ * (EQ - 1) // 2
    tau = int(args.tau) if args.tau is not None else int(math.ceil(EQ / 2.0))
    min_num_overlap_pairs = int(args.min_num_overlap_pairs) if args.min_num_overlap_pairs is not None else EQ
    min_distinct_intersections = (
        int(args.min_distinct_intersections) 
        if args.min_distinct_intersections is not None 
        else max(3, int(math.ceil(EQ / 2.0)))
    )
    min_distinct_sizes = int(args.min_distinct_sizes) if args.min_distinct_sizes is not None else max(2, int(math.ceil(EQ / 2.0)))
    strong_overlap_tau = int(args.strong_overlap_tau) if args.strong_overlap_tau is not None else tau + 1
    min_num_strong_overlap_pairs = (
        int(args.min_num_strong_overlap_pairs)
        if args.min_num_strong_overlap_pairs is not None
        else min(total_pairs, max(2, int(math.ceil(EQ / 2.0))))
    )

    rng = np.random.default_rng(SEED)

    B = load_npz(B_PATH).tocsr()
    V, E = B.shape
    edge_sizes = np.asarray(B.sum(axis=0)).ravel().astype(np.int32)

    print(f"Loaded B: path={B_PATH}, shape={B.shape}, nnz={B.nnz}", flush=True)
    print(
        f"Params: eq_target={EQ}, tau={tau}, min_num_overlap_pairs={min_num_overlap_pairs}, "
        f"min_distinct_intersections={min_distinct_intersections}, "
        f"min_distinct_sizes={min_distinct_sizes}, "
        f"strong_overlap_tau={strong_overlap_tau}, "
        f"min_num_strong_overlap_pairs={min_num_strong_overlap_pairs}, "
        f"U_MIN={U_MIN}, U_MAX={U_MAX}, MAX_TRIES={MAX_TRIES}",
        flush=True
    )

    print("Building pairwise intersection matrix C = B^T B ...", flush=True)
    C = (B.T @ B).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()

    print("Building tau-neighbor sets ...", flush=True)
    tau_neigh_sets, tau_degrees = build_neighbor_sets_from_C(C, tau)
    print(f"tau-degree min/avg/max = {tau_degrees.min()}/{tau_degrees.mean():.2f}/{tau_degrees.max()}", flush=True)

    print("Building strong-neighbor sets ...", flush=True)
    strong_neigh_sets, strong_degrees = build_neighbor_sets_from_C(C, strong_overlap_tau)
    print(f"strong-degree min/avg/max = {strong_degrees.min()}/{strong_degrees.mean():.2f}/{strong_degrees.max()}", flush=True)

    print("Building edge -> vertices map ...", flush=True)
    edge_vertices = edge_vertices_from_B(B)

    tries = 0
    saved = 0
    seen = set()

    rej = {
        "duplicate_query": 0,
        "frontier_empty": 0,
        "frontier_all_rejected_by_u": 0,
        "U_out_of_range": 0,
        "not_connected": 0,
        "not_all_pairs_overlap": 0,
        "overlap_pairs_too_few": 0,
        "distinct_intersections_too_few": 0,
        "distinct_sizes_too_few": 0,
        "strong_overlap_pairs_too_few": 0,
    }

    while saved < N_QUERIES and tries < MAX_TRIES:
        tries += 1
        if PROGRESS_EVERY > 0 and tries % PROGRESS_EVERY == 0:
            print(f"[progress] tries={tries}, saved={saved}", flush=True)

        result, reason = sample_one_query(
            rng=rng,
            tau_neigh_sets=tau_neigh_sets,
            strong_neigh_sets=strong_neigh_sets,
            tau_degrees=tau_degrees,
            edge_vertices=edge_vertices,
            edge_sizes=edge_sizes,
            C=C,
            eq_target=EQ,
            tau=tau,
            min_num_overlap_pairs=min_num_overlap_pairs,
            require_connected_overlap_graph=args.require_connected_overlap_graph,
            require_all_pairs_overlap=args.require_all_pairs_overlap,
            min_distinct_intersections=min_distinct_intersections,
            min_distinct_sizes=min_distinct_sizes,
            strong_overlap_tau=strong_overlap_tau,
            min_num_strong_overlap_pairs=min_num_strong_overlap_pairs,
            u_min=U_MIN,
            u_max=U_MAX,
        )

        if result is None:
            rej[reason] = rej.get(reason, 0) + 1
            continue

        chosen = result["chosen_edges"]
        key = tuple(sorted(int(x) for x in chosen))
        if key in seen:
            rej["duplicate_query"] += 1
            continue

        seen.add(key)
        saved += 1

        qname = f"q{saved:04d}"
        qdir = OUT_ROOT / qname
        qdir.mkdir(parents=True, exist_ok=True)

        involved_nodes = np.array(result["chosen_nodes"], dtype=np.int64)

        Q = B[:, chosen][involved_nodes, :].tocsr()
        save_npz(qdir / "Q_incidence_UxEq.npz", Q)

        Q_sub = B[:, chosen].tocsr()
        interQ = (Q_sub.T @ Q_sub).toarray().astype(np.int64)
        np.fill_diagonal(interQ, 0)
        np.save(qdir / "Q_intersections_EqxEq.npy", interQ)

        meta = {
            "family": "unified_structural_constraint_protocol",
            "mode": "eq_sweep_pilot",
            "eq_target": int(EQ),
            "tau": int(tau),
            "min_num_overlap_pairs": int(min_num_overlap_pairs),
            "require_connected_overlap_graph": bool(args.require_connected_overlap_graph),
            "require_all_pairs_overlap": bool(args.require_all_pairs_overlap),
            "min_distinct_intersections": int(min_distinct_intersections),
            "min_distinct_sizes": int(min_distinct_sizes),
            "strong_overlap_tau": int(strong_overlap_tau),
            "min_num_strong_overlap_pairs": int(min_num_strong_overlap_pairs),
            "U_minmax": [int(U_MIN), int(U_MAX)],
            "chosen_edges_in_data": [int(e) for e in chosen],
            "chosen_edge_sizes": result["meta"]["chosen_sizes"],
            "overlap_degrees": result["meta"]["overlap_degrees"],
            "num_overlap_pairs": int(result["meta"]["num_overlap_pairs"]),
            "distinct_intersections": result["meta"]["distinct_intersections"],
            "distinct_sizes": result["meta"]["distinct_sizes"],
            "strong_pairs": int(result["meta"]["strong_pairs"]),
            "U": int(result["meta"]["U"]),
            "bpath": str(B_PATH),
            "out_root": str(OUT_ROOT),
        }
        (qdir / "Q_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        (qdir / "q_edges.json").write_text(
            json.dumps({
                "chosen_edges_in_data": [int(e) for e in chosen],
                "involved_nodes_in_data": involved_nodes.tolist()
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(
            f"[saved {saved}/{N_QUERIES}] {qname}: "
            f"Eq={EQ}, U={result['meta']['U']}, "
            f"overlap_pairs={result['meta']['num_overlap_pairs']}, "
            f"strong_pairs={result['meta']['strong_pairs']}, "
            f"distinct_intersections={len(result['meta']['distinct_intersections'])}, "
            f"distinct_sizes={len(result['meta']['distinct_sizes'])}",
            flush=True
        )

    summary = {
        "family": "unified_structural_constraint_protocol",
        "mode": "eq_sweep_pilot",
        "eq_target": int(EQ),
        "tau": int(tau),
        "min_num_overlap_pairs": int(min_num_overlap_pairs),
        "min_distinct_intersections": int(min_distinct_intersections),
        "min_distinct_sizes": int(min_distinct_sizes),
        "strong_overlap_tau": int(strong_overlap_tau),
        "min_num_strong_overlap_pairs": int(min_num_strong_overlap_pairs),
        "tries": int(tries),
        "saved": int(saved),
        "U_minmax": [int(U_MIN), int(U_MAX)],
        "reject_stats": rej,
        "bpath": str(B_PATH),
        "out_root": str(OUT_ROOT),
    }

    (OUT_ROOT / "_pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n===== PILOT SUMMARY =====", flush=True)
    print(f"tries={tries}, saved={saved}, eq_target={EQ}", flush=True)
    print("reject stats:", rej, flush=True)
    print("summary file:", OUT_ROOT / "_pilot_summary.json", flush=True)


if __name__ == "__main__":
    main()