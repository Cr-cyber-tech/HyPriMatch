from pathlib import Path
import argparse
import subprocess
import time
import json
import re
import sys

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=r"E:\zyc\project\out\queries\queryset",
                    help="queryset root dir containing q0001,q0002,...")
    ap.add_argument("--py", type=str, default=sys.executable,
                    help="python executable")
    ap.add_argument("--src", type=str, default=r"E:\zyc\project\src",
                    help="src directory containing step scripts")
    ap.add_argument("--pattern", type=str, default=r"q\d{4}",
                    help="folder name regex under root")
    ap.add_argument("--start", type=int, default=1, help="start index (q0001)")
    ap.add_argument("--end", type=int, default=-1, help="end index inclusive, -1 means all")
    ap.add_argument("--limit", type=int, default=-1, help="max number of queries to run, -1 means no limit")
    ap.add_argument("--dry", action="store_true", help="print commands only, do not run")
    ap.add_argument("--stop_on_error", action="store_true", help="stop immediately when a query fails")
    ap.add_argument("--skip_done", action="store_true", help="skip qdir if final outputs already exist")
    ap.add_argument("--timeout", type=int, default=0, help="per-step timeout seconds, 0 means no timeout")
    ap.add_argument("--require_injective", action="store_true",
                    help="pass --require_injective to client_step4_prf_filter.py")
    ap.add_argument("--global_data_share_dir", type=str, default=None,
                    help="global dir storing shared data-hypergraph tensors")

    # ===== 新增：给 client_17_make_query_prf_tokens.py 用 =====
    ap.add_argument("--data_dir", type=str, default=r"E:\zyc\project\data\hc",
                    help="dataset dir for query token generation")
    ap.add_argument("--owner_prf_dir", type=str, default=r"E:\zyc\project\out\hc_owner\prf",
                    help="owner prf dir passed to step4 cloud scripts")
    ap.add_argument("--bpath", type=str, default=r"E:\zyc\project\out\hc_incidence_VxE.npz",
                    help="incidence matrix B path passed to 06_make_dense_shares.py")
    ap.add_argument("--node_names_file", type=str, default=None,
                    help="node names filename under data_dir")
    ap.add_argument("--node_labels_file", type=str, default="node-labels-house-committees.txt",
                    help="node labels filename under data_dir")
    ap.add_argument("--dataset_id", type=str, default="hc",
                    help="dataset id for PRF input, e.g. hc / ch")
    ap.add_argument("--epoch", type=str, default="2026-01-26",
                    help="epoch string for PRF input")

    # 对没有 node name 的数据集再开这个开关；HC 先不用
    ap.add_argument("--use_vertex_id_as_sid", action="store_true",
                    help="pass --use_vertex_id_as_sid to client_17")
    ap.add_argument("--vertex_id_base", type=int, default=1,
                    help="vertex id base when using vertex IDs as sid")

    return ap.parse_args()

def run_cmd(cmd, cwd=None, timeout=0, dry=False, log_fp=None):
    cmd_str = " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd])
    if log_fp:
        log_fp.write("\n$ " + cmd_str + "\n")
        log_fp.flush()
    print("$", cmd_str)

    if dry:
        return 0

    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=(timeout if timeout and timeout > 0 else None)
        )
        out = p.stdout
        if log_fp:
            log_fp.write(out)
            log_fp.flush()
        else:
            print(out, end="")
        return p.returncode
    except subprocess.TimeoutExpired as e:
        if log_fp:
            log_fp.write(f"\n[TIMEOUT] {e}\n")
            log_fp.flush()
        print("[TIMEOUT]", e)
        return 124

def load_json_if_exists(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def extract_step4_brief(final_obj):
    if not isinstance(final_obj, dict):
        return None

    all_stats = final_obj.get("all_solutions_stats", [])
    chosen_idx = final_obj.get("chosen_solution_index", None)
    chosen_stat = None
    if isinstance(chosen_idx, int) and isinstance(all_stats, list) and 0 <= chosen_idx < len(all_stats):
        chosen_stat = all_stats[chosen_idx]

    brief = {
        "scheme": final_obj.get("scheme", None),
        "good_solutions_count": len(final_obj.get("good_solution_indices", [])),
        "chosen_solution_index": chosen_idx,
        "chosen_sol_id": final_obj.get("chosen_sol_id", None),
        "require_injective": final_obj.get("require_injective", None),
        "chosen_stat": chosen_stat,
    }
    return brief

def main():
    args = parse_args()
    root = Path(args.root)
    src = Path(args.src)

    assert root.exists(), f"root not found: {root}"
    assert src.exists(), f"src not found: {src}"

    pat = re.compile(args.pattern)

    qdirs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and pat.fullmatch(d.name):
            m = re.fullmatch(r"q(\d{4})", d.name)
            if not m:
                continue
            idx = int(m.group(1))
            if idx < args.start:
                continue
            if args.end != -1 and idx > args.end:
                continue
            qdirs.append(d)

    if args.limit != -1:
        qdirs = qdirs[:args.limit]

    if not qdirs:
        print("No query dirs found under:", root)
        return

    steps = [
        "06_make_dense_shares.py",

        "cloudA_step12_onlycloud.py",
        "cloudB_step12_onlycloud.py",
        "run_step12_socket_pair.py",

        "client_make_alpha.py",
        "cloudA_step3_dense.py",
        "cloudB_step3_dense.py",
        "client_decrypt_step3_dense.py",

        "client_16_make_padded_pairs.py",
        "client_17_make_query_prf_tokens.py",
        "client_18_share_query_tokens.py",
        "client_19_make_step4_preproc.py",
        "run_step4_socket_pair.py",
        "client_step4_prf_filter.py",

        "23_collect_query_comm_metrics.py",
    ]

    summary_root = {
        "root": str(root),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pattern": args.pattern,
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "dataset_config": {
            "data_dir": args.data_dir,
            "owner_prf_dir": args.owner_prf_dir,
            "bpath": args.bpath,
            "node_names_file": args.node_names_file,
            "node_labels_file": args.node_labels_file,
            "dataset_id": args.dataset_id,
            "epoch": args.epoch,
            "use_vertex_id_as_sid": args.use_vertex_id_as_sid,
            "vertex_id_base": args.vertex_id_base,
        },
        "results": []
    }

    ok_cnt = 0
    fail_cnt = 0

    for qdir in qdirs:
        print("\n" + "=" * 80)
        print("RUN:", qdir)
        print("=" * 80)

        final_path = qdir / "online_step4_prf" / "final_after_prf.json"
        if args.skip_done and final_path.exists():
            print("[SKIP_DONE]", qdir.name, "already has", final_path)
            obj = load_json_if_exists(final_path)
            summary_root["results"].append({
                "q": qdir.name,
                "qdir": str(qdir),
                "status": "skipped_done",
                "final": str(final_path),
                "final_brief": extract_step4_brief(obj)
            })
            continue

        log_path = qdir / "_runner.log"
        with log_path.open("w", encoding="utf-8") as log_fp:
            log_fp.write(f"[runner] qdir={qdir}\n")
            log_fp.write(f"[runner] start={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_fp.flush()

            t_q0 = time.time()
            status = "ok"
            failed_step = None

            step_timings = []

            for s in steps:
                script_path = src / s
                if not script_path.exists():
                    status = "fail_missing_script"
                    failed_step = s
                    msg = f"[ERROR] missing script: {script_path}"
                    print(msg)
                    log_fp.write(msg + "\n")
                    break

                cmd = [args.py, str(script_path), "--qdir", str(qdir)]

                # step-specific extra args
                if s == "06_make_dense_shares.py":
                    cmd.extend(["--bpath", args.bpath])
                    cmd.extend(["--global_data_share_dir", args.global_data_share_dir])
                
                if s == "run_step12_socket_pair.py":
                    cmd.extend(["--global_data_share_dir", args.global_data_share_dir])
                    cmd.extend(["--chunk", "10000"])
                
                if s == "cloudA_step3_dense.py":
                    cmd.extend(["--global_data_share_dir", args.global_data_share_dir])
                
                if s == "cloudB_step3_dense.py":
                    cmd.extend(["--global_data_share_dir", args.global_data_share_dir])
                
                if s == "client_decrypt_step3_dense.py":
                    cmd.extend(["--global_data_share_dir", args.global_data_share_dir])
                    cmd.append("--save_full_all")

                if s == "client_17_make_query_prf_tokens.py":
                    cmd.extend([
                        "--data_dir", args.data_dir,
                        "--node_labels_file", args.node_labels_file,
                        "--dataset_id", args.dataset_id,
                        "--epoch", args.epoch,
                    ])
                    if (not args.use_vertex_id_as_sid) and args.node_names_file:
                        cmd.extend(["--node_names_file", args.node_names_file])
                    if args.use_vertex_id_as_sid:
                        cmd.append("--use_vertex_id_as_sid")
                        cmd.extend(["--vertex_id_base", str(args.vertex_id_base)])

                if s == "run_step4_socket_pair.py":
                    cmd.extend(["--owner_prf_dir", args.owner_prf_dir])

                if s == "client_step4_prf_filter.py" and args.require_injective:
                    cmd.append("--require_injective")

                t0 = time.time()
                rc = run_cmd(cmd, cwd=src, timeout=args.timeout, dry=args.dry, log_fp=log_fp)
                t1 = time.time()

                step_timings.append({
                    "step": s,
                    "seconds": t1 - t0,
                    "return_code": rc,
                })

                if rc != 0:
                    status = "fail"
                    failed_step = s
                    msg = f"[ERROR] step failed: {s}, rc={rc}"
                    print(msg)
                    log_fp.write(msg + "\n")
                    break

        wall_seconds = time.time() - t_q0

        postprocess_steps = {
            "23_collect_query_comm_metrics.py",
        }

        query_seconds = sum(
            x["seconds"] for x in step_timings
            if x["step"] not in postprocess_steps
        )

        postprocess_seconds = sum(
            x["seconds"] for x in step_timings
            if x["step"] in postprocess_steps
        )

        record = {
            "q": qdir.name,
            "qdir": str(qdir),
            "status": status,
            "failed_step": failed_step,

            # 论文中的 query latency 用这个，不包含 23_collect_query_comm_metrics.py
            "seconds": query_seconds,

            # 这个只是保留完整 wall-clock 时间，方便以后检查
            "seconds_with_postprocess": wall_seconds,
            "postprocess_seconds": postprocess_seconds,

            "step_timings": step_timings,
            "log": str(log_path),
        }

        comm_path = qdir / "comm_metrics.json"
        if comm_path.exists():
            record["comm_metrics"] = load_json_if_exists(comm_path)

        if status == "ok" and final_path.exists():
            obj = load_json_if_exists(final_path)
            record["final_brief"] = extract_step4_brief(obj)
            ok_cnt += 1
        else:
            fail_cnt += 1

        summary_root["results"].append(record)

        if status != "ok" and args.stop_on_error:
            print("[STOP_ON_ERROR] stopping at", qdir.name)
            break

    summary_root["counts"] = {
        "ok": ok_cnt,
        "fail": fail_cnt,
        "global_data_share_dir": args.global_data_share_dir,
        "total": len(summary_root["results"])
    }

    out_summary = root / "_summary.json"
    out_summary.write_text(json.dumps(summary_root, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved summary:", out_summary)
    print("Counts:", summary_root["counts"])

if __name__ == "__main__":
    main()