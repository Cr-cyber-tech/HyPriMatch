from pathlib import Path
import argparse
import json
import socket
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
    return ap.parse_args()


def load_P(owner_dir: Path, fallback: int) -> int:
    sp = owner_dir / "share_params.json"
    if sp.exists():
        return int(json.loads(sp.read_text(encoding="utf-8"))["P"])
    return int(fallback)


def main():
    args = parse_args()
    Q_DIR = Path(args.qdir)
    STEP4 = Q_DIR / "online_step4_prf"
    OWNER = Path(args.owner_prf_dir)

    pair_path = STEP4 / "padded_pairs.json"
    qB_path = STEP4 / "query_tok_B.npy"
    dB_path = OWNER / "data_tok_B.npy"
    preproc_meta_path = STEP4 / "step4_preproc_meta.json"

    if not pair_path.exists():
        raise FileNotFoundError(f"Missing {pair_path}")
    if not qB_path.exists():
        raise FileNotFoundError(f"Missing {qB_path}")
    if not dB_path.exists():
        raise FileNotFoundError(f"Missing {dB_path}")
    if not preproc_meta_path.exists():
        raise FileNotFoundError(f"Missing {preproc_meta_path}")

    P = load_P(OWNER, args.p)

    obj = json.loads(pair_path.read_text(encoding="utf-8"))
    if obj.get("format") != "multi":
        raise ValueError("padded_pairs.json must be multi format.")

    padded_all = obj["padded_pairs"]
    U = int(obj["U"])
    K = int(obj["K"])
    S = int(obj["S"])
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


    qB = np.load(qB_path).astype(np.int64) % P
    dB = np.load(dB_path).astype(np.int64) % P

    flat_u = np.tile(np.repeat(np.arange(U, dtype=np.int64), K), S)
    flat_v = np.array([v for s in range(S) for row in padded_all[s] for v in row], dtype=np.int64)

    if flat_v.shape[0] != total:
        raise ValueError(f"flat_v len mismatch: {flat_v.shape[0]} vs total {total}")

    socket_stats = {
        "role": "CloudB",
        "qdir": str(Q_DIR),
        "actual_recv_bytes": 0,
        "actual_send_bytes": 0,
        "logical_recv_bytes": 0,
        "logical_send_bytes": 0,
        "parts": int(parts),
        "messages": [],
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(1)

        print(f"[CloudB-Step4-Socket] listening on {args.host}:{args.port}", flush=True)

        conn, addr = srv.accept()
        print(f"[CloudB-Step4-Socket] connected by {addr}", flush=True)

        with conn:
            offset = 0

            for part in range(parts):
                n = min(CHUNK, total - offset)
                uu = flat_u[offset:offset + n]
                vv = flat_v[offset:offset + n]

                meta, arrays, recv_bytes = recv_packet(conn)
                socket_stats["actual_recv_bytes"] += int(recv_bytes)

                if meta.get("tag") != "cloudA_open_part":
                    raise ValueError(f"Unexpected packet tag: {meta.get('tag')}")

                if int(meta.get("part")) != part:
                    raise ValueError(f"Expected part {part}, got {meta.get('part')}")

                eA = arrays["eA"].astype(np.int64).astype(object) % P
                fA = arrays["fA"].astype(np.int64).astype(object) % P

                socket_stats["logical_recv_bytes"] += int(arrays["eA"].nbytes + arrays["fA"].nbytes)

                preB_path = STEP4 / f"step4_preproc_B_part{part}.npz"
                if not preB_path.exists():
                    raise FileNotFoundError(f"Missing {preB_path}")

                preB = np.load(preB_path)

                mB = preB["m"].astype(np.int64).astype(object) % P
                aB = preB["a"].astype(np.int64).astype(object) % P
                bB = preB["b"].astype(np.int64).astype(object) % P
                cB = preB["c"].astype(np.int64).astype(object) % P

                if len(mB) != n or len(eA) != n:
                    raise ValueError(f"Part {part} len mismatch.")

                q0B = qB[uu, 0].astype(object)
                q1B = qB[uu, 1].astype(object)
                d0b = dB[vv, 0].astype(object)
                d1b = dB[vv, 1].astype(object)

                d0B = (q0B - d0b) % P
                d1B = (q1B - d1b) % P
                xB = (d0B + c * d1B) % P

                eB = (xB - aB) % P
                fB = (mB - bB) % P

                e = (eA + eB) % P
                f = (fA + fB) % P

                e_arr = np.array(e, dtype=np.int64)
                f_arr = np.array(f, dtype=np.int64)

                # 保存 transcript，兼容原来的统计脚本
                np.savez_compressed(
                    STEP4 / f"public_ef_part{part}.npz",
                    e=e_arr,
                    f=f_arr,
                )

                # B 的输出 share
                zB = (cB + e * bB + f * aB + e * f) % P
                np.savez_compressed(
                    STEP4 / f"cloudB_z_part{part}.npz",
                    z=np.array(zB, dtype=np.int64),
                )

                send_bytes = send_packet(
                    conn,
                    meta={
                        "tag": "public_ef_part",
                        "part": int(part),
                        "n": int(n),
                        "src": "CloudB",
                        "dst": "CloudA",
                    },
                    arrays={
                        "e": e_arr,
                        "f": f_arr,
                    },
                )

                socket_stats["actual_send_bytes"] += int(send_bytes)
                socket_stats["logical_send_bytes"] += int(e_arr.nbytes + f_arr.nbytes)

                socket_stats["messages"].append({
                    "part": int(part),
                    "n": int(n),
                    "recv_logical_bytes": int(arrays["eA"].nbytes + arrays["fA"].nbytes),
                    "send_logical_bytes": int(e_arr.nbytes + f_arr.nbytes),
                    "recv_actual_socket_bytes": int(recv_bytes),
                    "send_actual_socket_bytes": int(send_bytes),
                })

                offset += n
                print(f"[CloudB-Step4-Socket] part={part}, n={n}", flush=True)

            meta, arrays, recv_bytes = recv_packet(conn)
            socket_stats["actual_recv_bytes"] += int(recv_bytes)
            if meta.get("tag") != "done":
                raise ValueError(f"Expected done packet, got {meta}")

    (STEP4 / "socket_cloudB_step4_stats.json").write_text(
        json.dumps(socket_stats, indent=2),
        encoding="utf-8",
    )

    print("[CloudB-Step4-Socket] saved:", STEP4 / "socket_cloudB_step4_stats.json", flush=True)


if __name__ == "__main__":
    main()