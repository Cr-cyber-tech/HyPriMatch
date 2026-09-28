from pathlib import Path
import argparse
import subprocess
import sys
import json
import csv
import time

PROJECT_ROOT = Path(r"E:\zyc\project")
PYTHON = sys.executable

DATASET_CONFIGS = {
    "hc": {
        "out_dir": PROJECT_ROOT / "out",
        "query_prefix": "pilot_protocol_hc_eq",
        "bpath": PROJECT_ROOT / "out" / "hc_incidence_VxE.npz",
        "data_dir": PROJECT_ROOT / "data" / "hc",
        "owner_prf_dir": PROJECT_ROOT / "out" / "hc_owner" / "prf",
        "global_data_share_dir": PROJECT_ROOT / "out" / "data_shares",
        "node_names_file": "node-names-house-committees.txt",
        "node_labels_file": "node-labels-house-committees.txt",
        "use_vertex_id_as_sid": False,
        "vertex_id_base": 1,
    },
    "ma": {
        "out_dir": PROJECT_ROOT / "out_ma",
        "query_prefix": "pilot_protocol_ma_eq",
        "bpath": PROJECT_ROOT / "out_ma" / "ma_incidence_VxE.npz",
        "data_dir": PROJECT_ROOT / "data" / "ma",
        "owner_prf_dir": PROJECT_ROOT / "out_ma" / "ma_owner" / "prf",
        "global_data_share_dir": PROJECT_ROOT / "out_ma" / "data_shares",
        "node_names_file": None,
        "node_labels_file": "node-labels-mathoverflow-answers.txt",
        "use_vertex_id_as_sid": True,
        "vertex_id_base": 1,
    },
    "wt": {
        "out_dir": PROJECT_ROOT / "out_wt",
        "query_prefix": "pilot_protocol_wt_eq",
        "bpath": PROJECT_ROOT / "out_wt" / "wt_incidence_VxE.npz",
        "data_dir": PROJECT_ROOT / "data" / "wt",
        "owner_prf_dir": PROJECT_ROOT / "out_wt" / "wt_owner" / "prf",
        "global_data_share_dir": PROJECT_ROOT / "out_wt" / "data_shares",
        "node_names_file": None,
        "node_labels_file": "node-labels-walmart-trips.txt",
        "use_vertex_id_as_sid": True,
        "vertex_id_base": 1,
    },
}


def run_cmd(cmd, log_fp=None):
    cmd_str = " ".join([f'"{x}"' if " " in str(x) else str(x) for x in cmd])
    print("\n$", cmd_str)

    if log_fp:
        log_fp.write("\n$ " + cmd_str + "\n")
        log_fp.flush()

    p = subprocess.run(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if log_fp:
        log_fp.write(p.stdout)
        log_fp.flush()

    print(p.stdout, end="")
    return p.returncode


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_hit_from_final(final_obj, qdir: Path):
    """
    Hit definition:
    1. final_after_prf.json exists;
    2. exactly one good solution;
    3. chosen solution exists and is valid;
    4. every query vertex has exactly one final candidate;
    5. recovered candidates equal q_edges.json/involved_nodes_in_data in order.
    """
    if not isinstance(final_obj, dict):
        return None, "missing_or_invalid_final_json"

    q_edges_path = qdir / "q_edges.json"
    q_edges = load_json(q_edges_path)
    if not isinstance(q_edges, dict):
        return None, "missing_q_edges_json"

    true_nodes = q_edges.get("involved_nodes_in_data", None)
    if not isinstance(true_nodes, list) or len(true_nodes) == 0:
        return None, "missing_involved_nodes_in_data"

    true_nodes = [int(x) for x in true_nodes]
    U = len(true_nodes)

    good = final_obj.get("good_solution_indices", [])
    if not isinstance(good, list):
        return 0, "bad_good_solution_indices"

    if len(good) != 1:
        return 0, f"good_solution_count_not_one:{len(good)}"

    chosen_idx = final_obj.get("chosen_solution_index", None)
    all_stats = final_obj.get("all_solutions_stats", [])

    if not isinstance(chosen_idx, int):
        return 0, "missing_chosen_solution_index"

    if not isinstance(all_stats, list) or not (0 <= chosen_idx < len(all_stats)):
        return 0, "chosen_solution_index_out_of_range"

    chosen_stat = all_stats[chosen_idx]
    if not isinstance(chosen_stat, dict):
        return 0, "invalid_chosen_stat"

    if chosen_stat.get("feasible_no_zero") is not True:
        return 0, "not_feasible_no_zero"

    if chosen_stat.get("injective_ok") is not True:
        return 0, "not_injective"

    if int(chosen_stat.get("num_zero", -1)) != 0:
        return 0, "num_zero_not_zero"

    if int(chosen_stat.get("num_unique", -1)) != U:
        return 0, f"num_unique_mismatch:{chosen_stat.get('num_unique')}!=U{U}"

    # Your final_after_prf.json stores chosen_final_candidates at top level.
    # Fallback to chosen_stat for compatibility with old outputs.
    cand = final_obj.get("chosen_final_candidates", None)
    if cand is None:
        cand = chosen_stat.get("chosen_final_candidates", None)

    if not isinstance(cand, list):
        return 0, "missing_chosen_final_candidates"

    if len(cand) != U:
        return 0, f"candidate_length_mismatch:{len(cand)}!=U{U}"

    pred_nodes = []
    for i, xs in enumerate(cand):
        if not isinstance(xs, list):
            return 0, f"candidate_not_list_at_u{i}"
        if len(xs) != 1:
            return 0, f"candidate_not_unique_at_u{i}:len={len(xs)}"
        pred_nodes.append(int(xs[0]))

    if pred_nodes != true_nodes:
        return 0, "recovered_nodes_not_equal_true_nodes"

    return 1, "unique_true_match"


def collect_one_root(dataset: str, eq: int, root: Path):
    summary_path = root / "_summary.json"
    summary = load_json(summary_path)

    rows = []
    if not isinstance(summary, dict):
        return rows

    for rec in summary.get("results", []):
        q = rec.get("q")
        qdir = Path(rec.get("qdir", root / str(q)))
        status = rec.get("status")
        seconds = float(rec.get("seconds", 0.0))

        final_path = qdir / "online_step4_prf" / "final_after_prf.json"
        final_obj = load_json(final_path)
        hit, hit_reason = infer_hit_from_final(final_obj, qdir)

        comm_path = qdir / "comm_metrics.json"
        comm = load_json(comm_path) or {}

        rows.append({
            "dataset": dataset,
            "eq": int(eq),
            "q": q,
            "status": status,
            "query_time_sec": seconds,
            "hit": hit,
            "hit_reason": hit_reason,
            "online_comm_bytes": int(comm.get("online_comm_bytes", 0)),
            "offline_comm_bytes": int(comm.get("offline_comm_bytes", 0)),
            "online_comm_MB": float(comm.get("online_comm_MB", 0.0)),
            "offline_comm_MB": float(comm.get("offline_comm_MB", 0.0)),
            "failed_step": rec.get("failed_step"),
            "qdir": str(qdir),
        })

    return rows


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_rows(dataset: str, rows):
    out = []

    ok_statuses = {"ok", "skipped_done"}
    eqs = sorted(set(int(r["eq"]) for r in rows))

    for eq in eqs:
        sub = [r for r in rows if int(r["eq"]) == eq]
        ok = [r for r in sub if r["status"] in ok_statuses]
        hit_values = [r["hit"] for r in sub if r["hit"] is not None]

        out.append({
            "dataset": dataset,
            "eq": eq,
            "num_queries": len(sub),
            "num_ok": len(ok),
            "num_fail": len(sub) - len(ok),
            "avg_query_time_sec": mean([r["query_time_sec"] for r in sub]),
            "accuracy": mean(hit_values),
            "avg_online_comm_MB": mean([r["online_comm_MB"] for r in sub]),
            "avg_offline_comm_MB": mean([r["offline_comm_MB"] for r in sub]),
        })

    hit_values_all = [r["hit"] for r in rows if r["hit"] is not None]
    ok_all = [r for r in rows if r["status"] in ok_statuses]

    out.append({
        "dataset": dataset,
        "eq": "Avg",
        "num_queries": len(rows),
        "num_ok": len(ok_all),
        "num_fail": len(rows) - len(ok_all),
        "avg_query_time_sec": mean([r["query_time_sec"] for r in rows]),
        "accuracy": mean(hit_values_all),
        "avg_online_comm_MB": mean([r["online_comm_MB"] for r in rows]),
        "avg_offline_comm_MB": mean([r["offline_comm_MB"] for r in rows]),
    })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hc", "ma", "wt"], required=True)
    ap.add_argument("--eqs", type=str, default="3,5,7,9")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-step timeout seconds, 0 means no timeout")
    ap.add_argument("--epoch", type=str, default="2026-01-26")
    ap.add_argument("--stop_on_error", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--require_injective", action="store_true")
    ap.add_argument("--pipeline_script", type=str, default="21_run_queryset_pipeline_socket.py",
                    help="pipeline runner script; default uses socket version")
    args = ap.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    eqs = [int(x.strip()) for x in args.eqs.split(",") if x.strip()]

    src = PROJECT_ROOT / "src"
    pipeline_path = src / args.pipeline_script

    out_metrics_dir = cfg["out_dir"] / "metrics"
    out_metrics_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_metrics_dir / f"run_{args.dataset}_all_eq.log"
    all_rows = []

    with log_path.open("w", encoding="utf-8") as log_fp:
        log_fp.write(f"[run_dataset_all_eq] dataset={args.dataset}\n")
        log_fp.write(f"[run_dataset_all_eq] eqs={eqs}\n")
        log_fp.write(f"[run_dataset_all_eq] pipeline_script={pipeline_path}\n")
        log_fp.write(f"[run_dataset_all_eq] time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fp.flush()

        for eq in eqs:
            root = cfg["out_dir"] / "queries" / f"{cfg['query_prefix']}{eq}"

            print("\n" + "=" * 80)
            print(f"DATASET={args.dataset}, EQ={eq}")
            print("=" * 80)

            cmd = [
                PYTHON,
                pipeline_path,
                "--root", root,
                "--bpath", cfg["bpath"],
                "--owner_prf_dir", cfg["owner_prf_dir"],
                "--global_data_share_dir", cfg["global_data_share_dir"],
                "--dataset_id", args.dataset,
                "--data_dir", cfg["data_dir"],
                "--node_labels_file", cfg["node_labels_file"],
                "--epoch", args.epoch,
                "--start", args.start,
                "--end", args.end,
                "--timeout", args.timeout,
            ]

            if cfg["node_names_file"]:
                cmd.extend(["--node_names_file", cfg["node_names_file"]])

            if cfg["use_vertex_id_as_sid"]:
                cmd.append("--use_vertex_id_as_sid")
                cmd.extend(["--vertex_id_base", cfg["vertex_id_base"]])

            if args.stop_on_error:
                cmd.append("--stop_on_error")

            if args.skip_done:
                cmd.append("--skip_done")

            if args.require_injective:
                cmd.append("--require_injective")

            rc = run_cmd(cmd, log_fp=log_fp)
            if rc != 0:
                print(f"[WARN] pipeline returned rc={rc} for dataset={args.dataset}, eq={eq}")

            # 24 directly reads _summary.json + final_after_prf.json + comm_metrics.json.
            # No need to run the old 22_export_runner_summary_csv.py here.
            rows = collect_one_root(args.dataset, eq, root)
            all_rows.extend(rows)

    per_query_csv = out_metrics_dir / f"{args.dataset}_per_query_metrics.csv"
    summary_csv = out_metrics_dir / f"{args.dataset}_summary_metrics.csv"

    write_csv(per_query_csv, all_rows)
    summary_rows = summarize_rows(args.dataset, all_rows)
    write_csv(summary_csv, summary_rows)

    print("\n" + "=" * 80)
    print("DATASET RUN FINISHED")
    print("=" * 80)
    print("Per-query metrics:", per_query_csv)
    print("Summary metrics  :", summary_csv)
    print("Log file         :", log_path)


if __name__ == "__main__":
    main()
