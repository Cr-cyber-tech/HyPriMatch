from pathlib import Path
import argparse
import subprocess
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True)
    ap.add_argument("--global_data_share_dir", type=str, required=True)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6212)
    ap.add_argument("--py", type=str, default=sys.executable)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--triple_seed", type=int, default=777)
    ap.add_argument("--eq_seed", type=int, default=20260202)
    ap.add_argument("--max_solutions", type=int, default=500)
    ap.add_argument("--allow_reuse", action="store_true")
    return ap.parse_args()


def tail_text(path: Path, n_lines: int = 80) -> str:
    if not path.exists():
        return f"[missing log] {path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception as e:
        return f"[failed to read log] {path}: {e}"


def terminate_process(p: subprocess.Popen, name: str):
    if p.poll() is None:
        print(f"[run-step12-socket] terminating {name}...")
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[run-step12-socket] killing {name}...")
            p.kill()
            p.wait()


def main():
    args = parse_args()
    src = Path(__file__).resolve().parent
    qdir = Path(args.qdir)
    out_dir = qdir / "online_step12_cloud"
    out_dir.mkdir(parents=True, exist_ok=True)

    cloudA_log = out_dir / "run_step12_socket_cloudA.log"
    cloudB_log = out_dir / "run_step12_socket_cloudB.log"

    cloudB = [
        args.py,
        str(src / "cloudB_step12_socket_server.py"),
        "--qdir", args.qdir,
        "--global_data_share_dir", args.global_data_share_dir,
        "--host", args.host,
        "--port", str(args.port),
        "--chunk", str(args.chunk),
        "--triple_seed", str(args.triple_seed),
        "--eq_seed", str(args.eq_seed),
    ]

    cloudA = [
        args.py,
        str(src / "cloudA_step12_socket_client.py"),
        "--qdir", args.qdir,
        "--global_data_share_dir", args.global_data_share_dir,
        "--host", args.host,
        "--port", str(args.port),
        "--chunk", str(args.chunk),
        "--triple_seed", str(args.triple_seed),
        "--eq_seed", str(args.eq_seed),
        "--max_solutions", str(args.max_solutions),
    ]

    if args.allow_reuse:
        cloudA.append("--allow_reuse")
        cloudB.append("--allow_reuse")

    print("[run-step12-socket] CloudA log:", cloudA_log)
    print("[run-step12-socket] CloudB log:", cloudB_log)

    with cloudB_log.open("w", encoding="utf-8", errors="ignore") as fb:
        print("[run-step12-socket] start CloudB server")
        pB = subprocess.Popen(
            cloudB,
            stdout=fb,
            stderr=subprocess.STDOUT,
            text=True,
        )

        time.sleep(1.0)

        with cloudA_log.open("w", encoding="utf-8", errors="ignore") as fa:
            print("[run-step12-socket] start CloudA client")
            pA = subprocess.run(
                cloudA,
                stdout=fa,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if pA.returncode != 0:
            terminate_process(pB, "CloudB")
            print(f"[run-step12-socket] CloudA failed rc={pA.returncode}")
            print("\n========== CloudA log tail ==========")
            print(tail_text(cloudA_log))
            print("\n========== CloudB log tail ==========")
            print(tail_text(cloudB_log))
            sys.exit(pA.returncode)

        try:
            rcB = pB.wait(timeout=300)
        except subprocess.TimeoutExpired:
            terminate_process(pB, "CloudB")
            print("[run-step12-socket] CloudB did not exit after CloudA finished.")
            print("\n========== CloudA log tail ==========")
            print(tail_text(cloudA_log))
            print("\n========== CloudB log tail ==========")
            print(tail_text(cloudB_log))
            sys.exit(124)

        if rcB != 0:
            print(f"[run-step12-socket] CloudB failed rc={rcB}")
            print("\n========== CloudA log tail ==========")
            print(tail_text(cloudA_log))
            print("\n========== CloudB log tail ==========")
            print(tail_text(cloudB_log))
            sys.exit(rcB)

    print("[run-step12-socket] OK")


if __name__ == "__main__":
    main()