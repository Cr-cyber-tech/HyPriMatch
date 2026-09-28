from pathlib import Path
import argparse
import json
import socket
import time
import numpy as np
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from socket_sim.net import send_packet, recv_packet

P_DEFAULT = 2305843009213693951


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True)
    ap.add_argument("--owner_prf_dir", type=str, required=True)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6204)
    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--connect_retry", type=int, default=30)
    ap.add_argument("--connect_sleep", type=float, default=0.5)
    return ap.parse_args()


def load_P(owner_dir: Path, fallback: int) -> int:
    sp = owner_dir / "share_params.json"
    if sp.exists():
        return int(json.loads(sp.read_text(encoding="utf-8"))["P"])
    return int(fallback)


def connect_with_retry(host, port, retry, sleep_s):
    last_err = None
    for _ in range(retry):
        try:
            conn = socket.create_connection((host, port), timeout=10)
            conn.settimeout(None)
            return conn
        except OSError as e:
            last_err = e
            time.sleep(sleep_s)
    raise ConnectionError(f"cannot connect to {host}:{port}: {last_err}")


def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)
    STEP4 = Q_DIR / "online_step4_prf"
    STEP4.mkdir(parents=True, exist_ok=True)

    OWNER = Path(args.owner_prf_dir)
    P = load_P(OWNER, args.p)

    pair_path = STEP4 / "padded_pairs.json"
    qA_path = STEP4 / "query_tok_A.npy"
    dA_path = OWNER / "data_tok_A.npy"
    preproc_meta_path = STEP4 / "step4_preproc_meta.json"

    if not pair_path.exists():
        raise FileNotFoundError(f"Missing {pair_path}")
    if not qA_path.exists():
        raise FileNotFoundError(f"Missing {qA_path}")
    if not dA_path.exists():
        raise FileNotFoundError(f"Missing {dA_path}")
    if not preproc_meta_path.exists():
        raise FileNotFoundError(f"Missing {preproc_meta_path}")

    obj = json.loads(pair_path.read_text(encoding="utf-8"))
    if obj.get("format") != "multi":
        raise ValueError("padded_pairs.json must be multi format with padded_pairs[S][U][K].")

    padded_all = obj["padded_pairs"]
    U = int(obj["U"])
    K = int(obj["K"])
    S = int(obj["S"])
    sol_ids = obj.get("sol_ids", list(range(S)))
    total = S * U * K

    pre_meta = json.loads(preproc_meta_path.read_text(encoding="utf-8"))
    CHUNK = int(pre_meta["chunk"])
    parts = int(pre_meta["parts"])

    if int(pre_meta["P"]) != P:
        raise ValueError(
            f"P mismatch: preproc P={pre_meta['P']}, owner P={P}"
        )

    if "merge_coefficient" not in pre_meta:
        raise ValueError(
            "Missing merge_coefficient in step4_preproc_meta.json. "
            "Please rerun the Step-4 preprocessing script."
        )

    c = int(pre_meta["merge_coefficient"]) % P

    qA = np.load(qA_path).astype(np.int64) % P
    dA = np.load(dA_path).astype(np.int64) % P


    flat_u = np.tile(np.repeat(np.arange(U, dtype=np.int64), K), S)
    flat_v = np.array([v for s in range(S) for row in padded_all[s] for v in row], dtype=np.int64)

    if flat_v.shape[0] != total:
        raise ValueError(f"flat_v len mismatch: {flat_v.shape[0]} vs total {total}")

    meta = {
        "scheme": "masked_zero_test_share_beaver_socket",
        "merge_limbs": True,
        "c": int(c),
        "P": int(P),
        "U": int(U),
        "K": int(K),
        "S": int(S),
        "sol_ids": sol_ids,
        "total": int(total),
        "chunk": int(CHUNK),
        "parts": int(parts),
        "qdir": str(Q_DIR),
        "owner_prf_dir": str(OWNER),
        "socket": {
            "CloudA": "client",
            "CloudB": "server",
            "host": args.host,
            "port": int(args.port),
        }
    }
    (STEP4 / "cloud_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    socket_stats = {
        "role": "CloudA",
        "qdir": str(Q_DIR),
        "actual_send_bytes": 0,
        "actual_recv_bytes": 0,
        "logical_send_bytes": 0,
        "logical_recv_bytes": 0,
        "parts": int(parts),
        "messages": [],
    }

    with connect_with_retry(args.host, args.port, args.connect_retry, args.connect_sleep) as conn:
        offset = 0

        for part in range(parts):
            n = min(CHUNK, total - offset)
            uu = flat_u[offset:offset + n]
            vv = flat_v[offset:offset + n]

            preA_path = STEP4 / f"step4_preproc_A_part{part}.npz"
            if not preA_path.exists():
                raise FileNotFoundError(f"Missing {preA_path}")

            preA = np.load(preA_path)

            mA = preA["m"].astype(np.int64).astype(object) % P
            aA = preA["a"].astype(np.int64).astype(object) % P
            bA = preA["b"].astype(np.int64).astype(object) % P
            cA = preA["c"].astype(np.int64).astype(object) % P

            if len(mA) != n:
                raise ValueError(f"Part {part} preproc A len mismatch: expected {n}, got {len(mA)}")

            q0A = qA[uu, 0].astype(object)
            q1A = qA[uu, 1].astype(object)
            d0a = dA[vv, 0].astype(object)
            d1a = dA[vv, 1].astype(object)

            d0A = (q0A - d0a) % P
            d1A = (q1A - d1a) % P
            xA = (d0A + c * d1A) % P

            eA = (xA - aA) % P
            fA = (mA - bA) % P

            eA_arr = np.array(eA, dtype=np.int64)
            fA_arr = np.array(fA, dtype=np.int64)

            # 保存本地状态，兼容原目录结构
            np.savez_compressed(
                STEP4 / f"cloudA_local_part{part}.npz",
                aA=np.array(aA, dtype=np.int64),
                bA=np.array(bA, dtype=np.int64),
                cA=np.array(cA, dtype=np.int64),
            )

            # 保存 transcript，不作为 B 的输入来源，只用于审计/兼容统计
            np.savez_compressed(
                STEP4 / f"cloudA_open_part{part}.npz",
                eA=eA_arr,
                fA=fA_arr,
            )

            send_bytes = send_packet(
                conn,
                meta={
                    "tag": "cloudA_open_part",
                    "part": int(part),
                    "n": int(n),
                    "src": "CloudA",
                    "dst": "CloudB",
                },
                arrays={
                    "eA": eA_arr,
                    "fA": fA_arr,
                },
            )

            socket_stats["actual_send_bytes"] += int(send_bytes)
            socket_stats["logical_send_bytes"] += int(eA_arr.nbytes + fA_arr.nbytes)

            meta_resp, arrays_resp, recv_bytes = recv_packet(conn)
            socket_stats["actual_recv_bytes"] += int(recv_bytes)

            if meta_resp.get("tag") != "public_ef_part":
                raise ValueError(f"Unexpected response tag: {meta_resp.get('tag')}")
            if int(meta_resp.get("part")) != part:
                raise ValueError(f"Expected response part {part}, got {meta_resp.get('part')}")

            e = arrays_resp["e"].astype(np.int64).astype(object) % P
            f = arrays_resp["f"].astype(np.int64).astype(object) % P

            socket_stats["logical_recv_bytes"] += int(arrays_resp["e"].nbytes + arrays_resp["f"].nbytes)

            zA = (cA + e * bA + f * aA) % P

            np.savez_compressed(
                STEP4 / f"cloudA_z_part{part}.npz",
                z=np.array(zA, dtype=np.int64),
            )

            socket_stats["messages"].append({
                "part": int(part),
                "n": int(n),
                "send_logical_bytes": int(eA_arr.nbytes + fA_arr.nbytes),
                "recv_logical_bytes": int(arrays_resp["e"].nbytes + arrays_resp["f"].nbytes),
                "send_actual_socket_bytes": int(send_bytes),
                "recv_actual_socket_bytes": int(recv_bytes),
            })

            offset += n
            print(f"[CloudA-Step4-Socket] part={part}, n={n}", flush=True)

        send_bytes = send_packet(
            conn,
            meta={
                "tag": "done",
                "src": "CloudA",
                "dst": "CloudB",
            },
            arrays={},
        )
        socket_stats["actual_send_bytes"] += int(send_bytes)

    (STEP4 / "socket_cloudA_step4_stats.json").write_text(
        json.dumps(socket_stats, indent=2),
        encoding="utf-8",
    )

    print("[CloudA-Step4-Socket] saved:", STEP4 / "socket_cloudA_step4_stats.json", flush=True)


if __name__ == "__main__":
    main()