from pathlib import Path
import argparse
import json
import socket
import time
import sys
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from socket_sim.net import send_packet, recv_packet

P_DEFAULT = 2305843009213693951


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True)
    ap.add_argument("--global_data_share_dir", type=str, required=True)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6212)
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--triple_seed", type=int, default=777)
    ap.add_argument("--eq_seed", type=int, default=20260202)
    ap.add_argument("--max_solutions", type=int, default=500)
    ap.add_argument("--allow_reuse", action="store_true")
    ap.add_argument("--connect_retry", type=int, default=30)
    ap.add_argument("--connect_sleep", type=float, default=0.5)
    return ap.parse_args()


def connect_with_retry(host, port, retry, sleep_s):
    last_err = None
    for _ in range(retry):
        try:
            conn = socket.create_connection((host, port), timeout=10)

            # 连接建立后改成阻塞模式。
            # 否则 create_connection 的 timeout=10 会继续作用在后续 recv 上，
            # WT Step2 较慢时 CloudA 会因为等待 CloudB 回包超过 10 秒而 TimeoutError。
            conn.settimeout(None)

            return conn
        except OSError as e:
            last_err = e
            time.sleep(sleep_s)
    raise ConnectionError(f"cannot connect to {host}:{port}: {last_err}")


def gen_eq_preproc_batch(P: int, seed: int, start_counter: int, n: int):
    seed_eff = (int(seed) + (int(start_counter) + 1) * 1000003) % (2**63 - 1)
    rng = np.random.default_rng(seed_eff)

    a = rng.integers(0, P, size=(n,), dtype=np.int64)
    b = rng.integers(0, P, size=(n,), dtype=np.int64)
    c = (a.astype(object) * b.astype(object)) % P
    c = np.array(c, dtype=np.int64)

    aA = rng.integers(0, P, size=(n,), dtype=np.int64)
    bA = rng.integers(0, P, size=(n,), dtype=np.int64)
    cA = rng.integers(0, P, size=(n,), dtype=np.int64)

    aB = (a.astype(object) - aA.astype(object)) % P
    bB = (b.astype(object) - bA.astype(object)) % P
    cB = (c.astype(object) - cA.astype(object)) % P

    r = rng.integers(1, P, size=(n,), dtype=np.int64)
    rA = rng.integers(0, P, size=(n,), dtype=np.int64)
    rB = (r.astype(object) - rA.astype(object)) % P

    return {
        "aA": np.array(aA, dtype=np.int64),
        "bA": np.array(bA, dtype=np.int64),
        "cA": np.array(cA, dtype=np.int64),
        "rA": np.array(rA, dtype=np.int64),
    }


class EqSocketClient:
    def __init__(self, conn, P, triple_seed):
        self.conn = conn
        self.P = int(P)
        self.triple_seed = int(triple_seed)
        self.counter = 0

        self.stats = {
            "actual_send_bytes": 0,
            "actual_recv_bytes": 0,
            "logical_send_bytes": 0,
            "logical_recv_bytes": 0,
            "eq_batches": 0,
            "eq_calls": 0,
            "step1_eq_calls": 0,
            "step2_eq_calls": 0,
            "messages": [],
        }

    def eq_batch(self, op, xA, extra_arrays):
        """
        xA is the secret-shared difference share on CloudA side.
        yA is random mask share rA.
        """
        P = self.P
        xA = np.asarray(xA, dtype=np.int64)
        n = int(xA.shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)

        counter_start = self.counter

        pre = gen_eq_preproc_batch(P, self.triple_seed, counter_start, n)
        aA = pre["aA"].astype(object) % P
        bA = pre["bA"].astype(object) % P
        cA = pre["cA"].astype(object) % P
        rA = pre["rA"].astype(object) % P

        xA_obj = xA.astype(object) % P
        yA = rA

        dA = (xA_obj - aA) % P
        eA = (yA - bA) % P

        dA_arr = np.array(dA, dtype=np.int64)
        eA_arr = np.array(eA, dtype=np.int64)

        arrays = dict(extra_arrays)
        arrays["dA"] = dA_arr
        arrays["eA"] = eA_arr

        send_bytes = send_packet(
            self.conn,
            meta={
                "tag": "eq_batch",
                "op": op,
                "counter_start": int(counter_start),
                "n": int(n),
                "src": "CloudA",
                "dst": "CloudB",
            },
            arrays=arrays,
        )

        self.stats["actual_send_bytes"] += int(send_bytes)
        self.stats["logical_send_bytes"] += int(dA_arr.nbytes + eA_arr.nbytes)

        meta_resp, arrays_resp, recv_bytes = recv_packet(self.conn)
        self.stats["actual_recv_bytes"] += int(recv_bytes)

        if meta_resp.get("tag") != "eq_batch_response":
            raise ValueError(f"Unexpected response tag: {meta_resp.get('tag')}")

        dB = arrays_resp["dB"].astype(np.int64).astype(object) % P
        eB = arrays_resp["eB"].astype(np.int64).astype(object) % P
        zB = arrays_resp["zB"].astype(np.int64).astype(object) % P

        self.stats["logical_recv_bytes"] += int(
            arrays_resp["dB"].nbytes + arrays_resp["eB"].nbytes + arrays_resp["zB"].nbytes
        )

        d = (dA + dB) % P
        e = (eA + eB) % P

        zA = (cA + d * bA + e * aA + d * e) % P
        zA_arr = np.array(zA, dtype=np.int64)

        z = (zA + zB) % P
        bits = (z == 0).astype(np.int64)

        send_z_bytes = send_packet(
            self.conn,
            meta={
                "tag": "zA_batch",
                "op": op,
                "counter_start": int(counter_start),
                "n": int(n),
                "src": "CloudA",
                "dst": "CloudB",
            },
            arrays={
                "zA": zA_arr,
            },
        )

        self.stats["actual_send_bytes"] += int(send_z_bytes)
        self.stats["logical_send_bytes"] += int(zA_arr.nbytes)

        self.counter += n
        self.stats["eq_batches"] += 1
        self.stats["eq_calls"] += int(n)

        if op == "step1":
            self.stats["step1_eq_calls"] += int(n)
        elif op == "step2":
            self.stats["step2_eq_calls"] += int(n)

        self.stats["messages"].append({
            "op": op,
            "counter_start": int(counter_start),
            "n": int(n),
            "send_logical_bytes_first": int(dA_arr.nbytes + eA_arr.nbytes),
            "recv_logical_bytes": int(arrays_resp["dB"].nbytes + arrays_resp["eB"].nbytes + arrays_resp["zB"].nbytes),
            "send_logical_bytes_zA": int(zA_arr.nbytes),
            "actual_send_bytes_first": int(send_bytes),
            "actual_recv_bytes": int(recv_bytes),
            "actual_send_bytes_zA": int(send_z_bytes),
        })

        return bits


def edge_sizes_candidates_socket(OUT, P, A_sizeB, A_sizeQ, eqc, chunk):
    E = int(A_sizeB.shape[0])
    Eq = int(A_sizeQ.shape[0])

    candidates = []

    for i in range(Eq):
        cand_i = []

        for start in range(0, E, chunk):
            end = min(E, start + chunk)
            e_idx = np.arange(start, end, dtype=np.int64)
            q_idx = np.full((end - start,), i, dtype=np.int64)

            xA = (A_sizeQ[q_idx].astype(object) - A_sizeB[e_idx].astype(object)) % P
            xA = np.array(xA, dtype=np.int64)

            bits = eqc.eq_batch(
                op="step1",
                xA=xA,
                extra_arrays={
                    "q_idx": q_idx,
                    "e_idx": e_idx,
                }
            )

            good = e_idx[bits == 1]
            cand_i.extend([int(x) for x in good.tolist()])

        candidates.append(cand_i)

    (OUT / "step1_candidates.json").write_text(
        json.dumps({
            "candidates": candidates,
            "sizes": [len(x) for x in candidates],
            "step1_eq_calls": int(eqc.stats["step1_eq_calls"]),
        }, indent=2),
        encoding="utf-8"
    )

    print("[CloudA-Step12-Socket] Step1 candidate sizes:", [len(x) for x in candidates], flush=True)
    return candidates


def solve_step2_socket(S, OUT, P, candidates, allow_reuse, max_solutions, global_data_share_dir, eqc):
    """
    Batched Step2 socket version.

    Compared with the previous version:
    - It still evaluates the same masked equality tests.
    - It still reveals only equality bits.
    - It batches multiple Step2 equality tests into one socket message.
    - It greatly reduces socket round trips on large datasets such as WT.
    """
    inter_A = np.load(S / "inter_A.npy")

    C_A_path = global_data_share_dir / "C_A.npy"
    if not C_A_path.exists():
        raise FileNotFoundError(f"Missing {C_A_path}")
    C_A = np.load(C_A_path, mmap_mode="r")

    Eq = len(candidates)

    if inter_A.shape != (Eq, Eq):
        raise ValueError(f"inter_A shape mismatch: {inter_A.shape}, Eq={Eq}")

    E = int(C_A.shape[0])
    order = sorted(range(Eq), key=lambda i: len(candidates[i]))

    cache_inter = {}
    cache_eq = {}

    # One socket message can contain up to this many Step2 equality tests.
    # 8192 is conservative and stable on Windows. You can raise it to 20000 later if needed.
    STEP2_EQ_BATCH = 8192

    dfs_nodes = 0
    batch_nodes = 0

    def get_inter_A(ea, eb):
        a = int(ea)
        b = int(eb)

        if a < 0 or b < 0 or a >= E or b >= E:
            raise IndexError(f"data edge index out of range: ({a}, {b}) with E={E}")

        if a <= b:
            x, y = a, b
        else:
            x, y = b, a

        key = (x, y)
        if key in cache_inter:
            return cache_inter[key]

        IA = int(C_A[x, y]) % P
        cache_inter[key] = IA
        return IA

    def make_eq_key(qi, qj, ei, ej):
        """
        Canonicalize one query-pair / data-edge-pair equality test.

        Query pair is stored as i < j.
        Data edge pair is also canonicalized because C is symmetric.
        """
        qi = int(qi)
        qj = int(qj)
        ei = int(ei)
        ej = int(ej)

        if qi <= qj:
            i, j = qi, qj
        else:
            i, j = qj, qi

        if ei <= ej:
            ea, eb = ei, ej
        else:
            ea, eb = ej, ei

        return (i, j, ea, eb)

    def batch_eval_missing_keys(missing_keys):
        """
        Evaluate all missing equality keys by batched socket calls.
        Results are written into cache_eq.
        """
        if not missing_keys:
            return

        # Deduplicate while preserving order.
        unique_keys = []
        seen = set()
        for key in missing_keys:
            if key not in seen and key not in cache_eq:
                seen.add(key)
                unique_keys.append(key)

        if not unique_keys:
            return

        for start in range(0, len(unique_keys), STEP2_EQ_BATCH):
            chunk_keys = unique_keys[start:start + STEP2_EQ_BATCH]

            i_idx = np.array([k[0] for k in chunk_keys], dtype=np.int64)
            j_idx = np.array([k[1] for k in chunk_keys], dtype=np.int64)
            ea_idx = np.array([k[2] for k in chunk_keys], dtype=np.int64)
            eb_idx = np.array([k[3] for k in chunk_keys], dtype=np.int64)

            xA_list = []
            for i, j, ea, eb in chunk_keys:
                IA = get_inter_A(ea, eb)
                tA = int(inter_A[i, j]) % P
                xA_list.append((IA - tA) % P)

            xA = np.array(xA_list, dtype=np.int64)

            bits = eqc.eq_batch(
                op="step2",
                xA=xA,
                extra_arrays={
                    "i_idx": i_idx,
                    "j_idx": j_idx,
                    "ea_idx": ea_idx,
                    "eb_idx": eb_idx,
                }
            )

            for key, bit in zip(chunk_keys, bits.tolist()):
                cache_eq[key] = bool(int(bit) == 1)

    def filter_candidates_for_next_edge(qi, assign, used):
        """
        For the next query edge qi, batch-check all candidate data edges
        against all previously assigned query edges.

        This replaces the old per-pair ok_pair() calls.
        """
        prev_qs = sorted(assign.keys())

        # If no previous query edge has been assigned, all unused candidates pass.
        if not prev_qs:
            out = []
            for e in candidates[qi]:
                e = int(e)
                if (not allow_reuse) and (e in used):
                    continue
                out.append(e)
            return out

        candidate_infos = []
        missing_keys = []

        for e in candidates[qi]:
            e = int(e)

            if (not allow_reuse) and (e in used):
                continue

            keys_for_e = []
            cached_failed = False

            for qj in prev_qs:
                ej = int(assign[qj])
                key = make_eq_key(qi, qj, e, ej)
                keys_for_e.append(key)

                if key in cache_eq:
                    if not cache_eq[key]:
                        cached_failed = True
                        break
                else:
                    missing_keys.append(key)

            if not cached_failed:
                candidate_infos.append((e, keys_for_e))

        # Batch-evaluate all uncached equality tests needed at this DFS node.
        batch_eval_missing_keys(missing_keys)

        passed = []
        for e, keys_for_e in candidate_infos:
            ok = True
            for key in keys_for_e:
                if not cache_eq.get(key, False):
                    ok = False
                    break

            if ok:
                passed.append(e)

        return passed

    solutions = []

    def reached_limit():
        return (max_solutions > 0) and (len(solutions) >= max_solutions)

    def dfs(pos, assign, used):
        nonlocal dfs_nodes, batch_nodes

        if reached_limit():
            return

        dfs_nodes += 1

        if pos == Eq:
            sol = [int(assign[i]) for i in range(Eq)]
            solutions.append(sol)
            return

        qi = order[pos]

        passed_candidates = filter_candidates_for_next_edge(qi, assign, used)
        batch_nodes += 1

        if batch_nodes % 100 == 0:
            print(
                f"[CloudA-Step12-Socket] Step2 progress: "
                f"dfs_nodes={dfs_nodes}, batch_nodes={batch_nodes}, "
                f"solutions={len(solutions)}, "
                f"step2_eq_calls={eqc.stats['step2_eq_calls']}, "
                f"cache_eq={len(cache_eq)}",
                flush=True
            )

        for e in passed_candidates:
            e = int(e)

            assign[qi] = e

            if not allow_reuse:
                used.add(e)

            dfs(pos + 1, assign, used)

            if not allow_reuse:
                used.remove(e)

            del assign[qi]

            if reached_limit():
                return

    dfs(0, {}, set())

    if len(solutions) == 0:
        raise RuntimeError("Step2 failed: no assignment satisfies all upper-triangle intersection constraints.")

    payload = {
        "format": "multi",
        "Eq": int(Eq),
        "num_solutions": int(len(solutions)),
        "max_solutions": int(max_solutions),
        "allow_reuse": bool(allow_reuse),
        "note": "Step2 batched socket version. CloudA drives DFS and batches masked equality tests with CloudB.",
        "matched_edges_solutions": solutions,
        "step2_eq_calls": int(eqc.stats["step2_eq_calls"]),
        "cache_inter_size": int(len(cache_inter)),
        "cache_eq_size": int(len(cache_eq)),
        "dfs_nodes": int(dfs_nodes),
        "batch_nodes": int(batch_nodes),
        "step2_eq_batch_size": int(STEP2_EQ_BATCH),
    }

    (OUT / "matched_edges.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )

    print(f"[CloudA-Step12-Socket] Step2 found {len(solutions)} solution(s).", flush=True)
    print("[CloudA-Step12-Socket] first solutions:", solutions[:min(5, len(solutions))], flush=True)
    print(
        f"[CloudA-Step12-Socket] Step2 stats: "
        f"dfs_nodes={dfs_nodes}, batch_nodes={batch_nodes}, "
        f"step2_eq_calls={eqc.stats['step2_eq_calls']}, "
        f"cache_eq={len(cache_eq)}",
        flush=True
    )

    return {
        "step2_eq_calls": int(eqc.stats["step2_eq_calls"]),
        "cache_inter_size": int(len(cache_inter)),
        "num_solutions": int(len(solutions)),
    }


def main():
    args = parse_args()

    Q_DIR = Path(args.qdir)
    P = int(args.p)

    S = Q_DIR / "shares_dense"
    OUT = Q_DIR / "online_step12_cloud"
    OUT.mkdir(parents=True, exist_ok=True)

    global_data_share_dir = Path(args.global_data_share_dir)

    A_sizeB = np.load(OUT / "A_sizeB.npy")
    A_sizeQ = np.load(OUT / "A_sizeQ.npy")

    with connect_with_retry(args.host, args.port, args.connect_retry, args.connect_sleep) as conn:
        eqc = EqSocketClient(conn, P=P, triple_seed=int(args.triple_seed))

        candidates = edge_sizes_candidates_socket(
            OUT=OUT,
            P=P,
            A_sizeB=A_sizeB,
            A_sizeQ=A_sizeQ,
            eqc=eqc,
            chunk=int(args.chunk),
        )

        step2_stats = solve_step2_socket(
            S=S,
            OUT=OUT,
            P=P,
            candidates=candidates,
            allow_reuse=bool(args.allow_reuse),
            max_solutions=int(args.max_solutions),
            global_data_share_dir=global_data_share_dir,
            eqc=eqc,
        )

        send_packet(
            conn,
            meta={
                "tag": "done",
                "src": "CloudA",
                "dst": "CloudB",
            },
            arrays={},
        )

    # logical communication metrics
    BYTES_PER_FIELD = 8
    ONLINE_BYTES_PER_EQ = 6 * BYTES_PER_FIELD
    OFFLINE_BYTES_PER_EQ = 8 * BYTES_PER_FIELD

    step1_eq_calls = int(eqc.stats["step1_eq_calls"])
    step2_eq_calls = int(eqc.stats["step2_eq_calls"])
    total_eq_calls = int(step1_eq_calls + step2_eq_calls)

    comm_metrics = {
        "qid": Q_DIR.name,
        "phase": "step12_socket",
        "step1_eq_calls": int(step1_eq_calls),
        "step2_eq_calls": int(step2_eq_calls),
        "total_eq_calls": int(total_eq_calls),

        "online_bytes_per_eq": int(ONLINE_BYTES_PER_EQ),
        "offline_bytes_per_eq": int(OFFLINE_BYTES_PER_EQ),

        "online_comm_bytes": int(total_eq_calls * ONLINE_BYTES_PER_EQ),
        "offline_comm_bytes": int(total_eq_calls * OFFLINE_BYTES_PER_EQ),

        "cache_inter_size": int(step2_stats["cache_inter_size"]),
        "num_solutions": int(step2_stats["num_solutions"]),

        "socket_actual_send_bytes_cloudA": int(eqc.stats["actual_send_bytes"]),
        "socket_actual_recv_bytes_cloudA": int(eqc.stats["actual_recv_bytes"]),
        "socket_logical_send_bytes_cloudA": int(eqc.stats["logical_send_bytes"]),
        "socket_logical_recv_bytes_cloudA": int(eqc.stats["logical_recv_bytes"]),

        "online_comm_note": "Socket Step12. Each equality exchanges dA,eA,zA and dB,eB,zB.",
        "offline_comm_note": "Each equality consumes one scalar Beaver triple and one random mask share pair.",
        "offline_material_type": "scalar arithmetic Beaver triple over mod P plus secret random mask r",
    }

    (OUT / "comm_metrics_step12.json").write_text(
        json.dumps(comm_metrics, indent=2),
        encoding="utf-8"
    )

    (OUT / "socket_cloudA_step12_stats.json").write_text(
        json.dumps({
            "role": "CloudA",
            "qdir": str(Q_DIR),
            **eqc.stats,
        }, indent=2),
        encoding="utf-8"
    )

    print("[CloudA-Step12-Socket] saved:", OUT / "comm_metrics_step12.json", flush=True)
    print("[CloudA-Step12-Socket] online_comm_MB =", comm_metrics["online_comm_bytes"] / 1024 / 1024, flush=True)
    print("[CloudA-Step12-Socket] offline_comm_MB =", comm_metrics["offline_comm_bytes"] / 1024 / 1024, flush=True)


if __name__ == "__main__":
    main()