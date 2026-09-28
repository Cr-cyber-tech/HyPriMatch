from pathlib import Path
import argparse
import json
import socket
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
    ap.add_argument("--allow_reuse", action="store_true")
    return ap.parse_args()


def gen_eq_preproc_batch(P: int, seed: int, start_counter: int, n: int):
    """
    Deterministically generate offline material for n masked equalities.

    Each equality uses:
      Beaver triple shares: aA,aB,bA,bB,cA,cB
      random mask shares: rA,rB
    """
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
        "aB": np.array(aB, dtype=np.int64),
        "bB": np.array(bB, dtype=np.int64),
        "cB": np.array(cB, dtype=np.int64),
        "rB": np.array(rB, dtype=np.int64),
    }


def main():
    args = parse_args()

    Q_DIR = Path(args.qdir)
    P = int(args.p)

    S = Q_DIR / "shares_dense"
    OUT = Q_DIR / "online_step12_cloud"
    OUT.mkdir(parents=True, exist_ok=True)

    global_data_share_dir = Path(args.global_data_share_dir)

    B_sizeB = np.load(OUT / "B_sizeB.npy")
    B_sizeQ = np.load(OUT / "B_sizeQ.npy")

    inter_B = np.load(S / "inter_B.npy")

    C_B_path = global_data_share_dir / "C_B.npy"
    if not C_B_path.exists():
        raise FileNotFoundError(f"Missing {C_B_path}")
    C_B = np.load(C_B_path, mmap_mode="r")

    stats = {
        "role": "CloudB",
        "qdir": str(Q_DIR),
        "actual_recv_bytes": 0,
        "actual_send_bytes": 0,
        "logical_recv_bytes": 0,
        "logical_send_bytes": 0,
        "eq_batches": 0,
        "eq_calls": 0,
        "messages": [],
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(1)

        print(f"[CloudB-Step12-Socket] listening on {args.host}:{args.port}", flush=True)

        conn, addr = srv.accept()
        print(f"[CloudB-Step12-Socket] connected by {addr}", flush=True)

        with conn:
            while True:
                meta, arrays, recv_bytes = recv_packet(conn)
                stats["actual_recv_bytes"] += int(recv_bytes)

                tag = meta.get("tag")

                if tag == "done":
                    print("[CloudB-Step12-Socket] received done", flush=True)
                    break

                if tag != "eq_batch":
                    raise ValueError(f"Unexpected packet tag: {tag}")

                op = meta.get("op")
                counter_start = int(meta["counter_start"])
                n = int(meta["n"])

                dA = arrays["dA"].astype(np.int64).astype(object) % P
                eA = arrays["eA"].astype(np.int64).astype(object) % P

                stats["logical_recv_bytes"] += int(arrays["dA"].nbytes + arrays["eA"].nbytes)

                pre = gen_eq_preproc_batch(P, int(args.triple_seed), counter_start, n)
                aB = pre["aB"].astype(object) % P
                bB = pre["bB"].astype(object) % P
                cB = pre["cB"].astype(object) % P
                rB = pre["rB"].astype(object) % P

                if op == "step1":
                    q_idx = arrays["q_idx"].astype(np.int64)
                    e_idx = arrays["e_idx"].astype(np.int64)

                    xB = (B_sizeQ[q_idx].astype(object) - B_sizeB[e_idx].astype(object)) % P

                elif op == "step2":
                    i_idx = arrays["i_idx"].astype(np.int64)
                    j_idx = arrays["j_idx"].astype(np.int64)
                    ea_idx = arrays["ea_idx"].astype(np.int64)
                    eb_idx = arrays["eb_idx"].astype(np.int64)

                    xB_list = []
                    for i, j, ea, eb in zip(i_idx.tolist(), j_idx.tolist(), ea_idx.tolist(), eb_idx.tolist()):
                        a = int(ea)
                        b = int(eb)
                        if a <= b:
                            x, y = a, b
                        else:
                            x, y = b, a

                        IB = int(C_B[x, y]) % P
                        tB = int(inter_B[int(i), int(j)]) % P
                        xB_list.append((IB - tB) % P)

                    xB = np.array(xB_list, dtype=np.int64).astype(object) % P

                else:
                    raise ValueError(f"Unknown op: {op}")

                yB = rB

                dB = (xB - aB) % P
                eB = (yB - bB) % P

                d = (dA + dB) % P
                e = (eA + eB) % P

                zB = (cB + d * bB + e * aB) % P

                dB_arr = np.array(dB, dtype=np.int64)
                eB_arr = np.array(eB, dtype=np.int64)
                zB_arr = np.array(zB, dtype=np.int64)

                send_bytes = send_packet(
                    conn,
                    meta={
                        "tag": "eq_batch_response",
                        "op": op,
                        "counter_start": counter_start,
                        "n": n,
                        "src": "CloudB",
                        "dst": "CloudA",
                    },
                    arrays={
                        "dB": dB_arr,
                        "eB": eB_arr,
                        "zB": zB_arr,
                    },
                )

                stats["actual_send_bytes"] += int(send_bytes)
                stats["logical_send_bytes"] += int(dB_arr.nbytes + eB_arr.nbytes + zB_arr.nbytes)

                # receive zA so CloudB can also reconstruct opened z
                meta_z, arrays_z, recv_z_bytes = recv_packet(conn)
                stats["actual_recv_bytes"] += int(recv_z_bytes)

                if meta_z.get("tag") != "zA_batch":
                    raise ValueError(f"Expected zA_batch, got {meta_z.get('tag')}")

                zA = arrays_z["zA"].astype(np.int64).astype(object) % P
                stats["logical_recv_bytes"] += int(arrays_z["zA"].nbytes)

                # CloudB can reconstruct z if needed
                _z = (zA + zB) % P

                stats["eq_batches"] += 1
                stats["eq_calls"] += int(n)
                stats["messages"].append({
                    "op": op,
                    "counter_start": counter_start,
                    "n": n,
                    "recv_logical_bytes_first": int(arrays["dA"].nbytes + arrays["eA"].nbytes),
                    "send_logical_bytes": int(dB_arr.nbytes + eB_arr.nbytes + zB_arr.nbytes),
                    "recv_logical_bytes_zA": int(arrays_z["zA"].nbytes),
                    "actual_recv_bytes_first": int(recv_bytes),
                    "actual_send_bytes": int(send_bytes),
                    "actual_recv_bytes_zA": int(recv_z_bytes),
                })

                if stats["eq_batches"] % 5000 == 0:
                    print(f"[CloudB-Step12-Socket] batches={stats['eq_batches']}, eq_calls={stats['eq_calls']}", flush=True)

    out_stats = OUT / "socket_cloudB_step12_stats.json"
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("[CloudB-Step12-Socket] saved:", out_stats, flush=True)


if __name__ == "__main__":
    main()