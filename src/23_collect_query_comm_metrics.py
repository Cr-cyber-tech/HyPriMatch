from pathlib import Path
import argparse
import json
import numpy as np


def npy_logical_nbytes(path: Path) -> int:
    arr = np.load(path, mmap_mode="r")
    return int(arr.nbytes)


def npz_logical_nbytes(path: Path) -> int:
    total = 0
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            total += int(z[k].nbytes)
    return int(total)


def file_logical_nbytes(path: Path) -> int:
    if path.suffix.lower() == ".npy":
        return npy_logical_nbytes(path)
    if path.suffix.lower() == ".npz":
        return npz_logical_nbytes(path)
    return int(path.stat().st_size)


def glob_sum(root: Path, pattern: str):
    """Sum logical bytes of matching npy/npz files, with details for audit."""
    total = 0
    details = []
    for p in sorted(root.glob(pattern)):
        if p.exists():
            b = file_logical_nbytes(p)
            total += b
            details.append({"file": str(p), "bytes": int(b)})
    return int(total), details


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def socket_actual_bytes_from_cloudA_stats(path: Path) -> int:
    obj = load_json(path)
    if not isinstance(obj, dict) or not obj:
        return 0
    return int(obj.get("actual_send_bytes", 0)) + int(obj.get("actual_recv_bytes", 0))


def socket_logical_bytes_from_cloudA_stats(path: Path) -> int:
    obj = load_json(path)
    if not isinstance(obj, dict) or not obj:
        return 0
    return int(obj.get("logical_send_bytes", 0)) + int(obj.get("logical_recv_bytes", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True)
    args = ap.parse_args()

    qdir = Path(args.qdir)

    # =========================
    # Step12: socket online + offline materials
    # =========================
    step12_dir = qdir / "online_step12_cloud"
    step12_path = step12_dir / "comm_metrics_step12.json"
    step12 = load_json(step12_path)

    if not isinstance(step12, dict) or not step12:
        raise FileNotFoundError(f"Missing or invalid Step12 metrics: {step12_path}")

    # Online communication uses actual socket bytes from CloudA side.
    # CloudA side includes both directions once, so it avoids double counting.
    step12_socket_online = 0
    if "socket_actual_send_bytes_cloudA" in step12 and "socket_actual_recv_bytes_cloudA" in step12:
        step12_socket_online = (
            int(step12.get("socket_actual_send_bytes_cloudA", 0))
            + int(step12.get("socket_actual_recv_bytes_cloudA", 0))
        )
    else:
        step12_socket_online = socket_actual_bytes_from_cloudA_stats(
            step12_dir / "socket_cloudA_step12_stats.json"
        )

    step12_socket_payload = 0
    if "socket_logical_send_bytes_cloudA" in step12 and "socket_logical_recv_bytes_cloudA" in step12:
        step12_socket_payload = (
            int(step12.get("socket_logical_send_bytes_cloudA", 0))
            + int(step12.get("socket_logical_recv_bytes_cloudA", 0))
        )
    else:
        step12_socket_payload = socket_logical_bytes_from_cloudA_stats(
            step12_dir / "socket_cloudA_step12_stats.json"
        )

    # Offline communication/material cost is actual offline material consumed.
    # For Step12 this is produced by cloudA_step12_socket_client.py as offline_comm_bytes.
    step12_offline = int(step12.get("offline_comm_bytes", 0))

    # =========================
    # Step4: socket online + offline materials
    # =========================
    step4_dir = qdir / "online_step4_prf"
    step4_socket_stats_path = step4_dir / "socket_cloudA_step4_stats.json"

    step4_socket_online = socket_actual_bytes_from_cloudA_stats(step4_socket_stats_path)
    step4_socket_payload = socket_logical_bytes_from_cloudA_stats(step4_socket_stats_path)

    # Step4 offline material actually used by PRF equality preprocessing.
    # We count logical array bytes inside the npz files, not compressed file size.
    preproc_A_bytes, preproc_A_details = glob_sum(step4_dir, "step4_preproc_A_part*.npz")
    preproc_B_bytes, preproc_B_details = glob_sum(step4_dir, "step4_preproc_B_part*.npz")
    step4_offline = int(preproc_A_bytes + preproc_B_bytes)

    # =========================
    # Final totals used by summary table
    # =========================
    online_total = int(step12_socket_online + step4_socket_online)
    offline_total = int(step12_offline + step4_offline)

    payload_total = int(step12_socket_payload + step4_socket_payload)

    metrics = {
        "q": qdir.name,
        "qdir": str(qdir),

        # Step12 details
        "step12_eq_calls": int(step12.get("total_eq_calls", 0)),
        "step12_step1_eq_calls": int(step12.get("step1_eq_calls", 0)),
        "step12_step2_eq_calls": int(step12.get("step2_eq_calls", 0)),
        "step12_online_comm_bytes": int(step12_socket_online),
        "step12_online_payload_bytes": int(step12_socket_payload),
        "step12_offline_comm_bytes": int(step12_offline),

        # Step4 details
        "step4_online_comm_bytes": int(step4_socket_online),
        "step4_online_payload_bytes": int(step4_socket_payload),
        "step4_offline_comm_bytes": int(step4_offline),

        # Final fields used by 24_run_dataset_all_eq.py
        # online_comm_MB = actual socket bytes exchanged by CloudA/CloudB.
        # offline_comm_MB = actual offline material consumed.
        "online_comm_bytes": int(online_total),
        "offline_comm_bytes": int(offline_total),
        "online_comm_MB": online_total / (1024 * 1024),
        "offline_comm_MB": offline_total / (1024 * 1024),

        # Optional audit field: socket payload bytes without JSON headers.
        # This is not used by the final summary table unless you choose to use it.
        "online_payload_bytes": int(payload_total),
        "online_payload_MB": payload_total / (1024 * 1024),

        "counting_policy": {
            "online_comm_MB": (
                "Actual socket bytes exchanged between CloudA and CloudB during online query processing, "
                "including Step12 socket messages and Step4 socket messages."
            ),
            "offline_comm_MB": (
                "Offline materials actually consumed during query processing, including Step12 Beaver triples/random masks "
                "and Step4 PRF equality preprocessing materials."
            ),
            "excluded": [
                "data upload shares under data_shares",
                "owner PRF token upload",
                "query upload shares under shares_dense",
                "query raw files and metadata",
            ],
        },

        "details": {
            "step12_metrics_file": str(step12_path),
            "step12_socket_stats_file": str(step12_dir / "socket_cloudA_step12_stats.json"),
            "step4_socket_stats_file": str(step4_socket_stats_path),
            "step4_preproc_A": preproc_A_details,
            "step4_preproc_B": preproc_B_details,
        },
    }

    out = qdir / "comm_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved:", out)
    print("online_comm_MB =", metrics["online_comm_MB"])
    print("offline_comm_MB =", metrics["offline_comm_MB"])


if __name__ == "__main__":
    main()
