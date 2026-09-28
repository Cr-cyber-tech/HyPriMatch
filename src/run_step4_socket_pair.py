from pathlib import Path
import argparse
import subprocess
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True)
    ap.add_argument("--owner_prf_dir", type=str, required=True)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6204)
    ap.add_argument("--py", type=str, default=sys.executable)
    return ap.parse_args()


def main():
    args = parse_args()
    src = Path(__file__).resolve().parent

    cloudB = [
        args.py,
        str(src / "cloudB_step4_socket_server.py"),
        "--qdir", args.qdir,
        "--owner_prf_dir", args.owner_prf_dir,
        "--host", args.host,
        "--port", str(args.port),
    ]

    cloudA = [
        args.py,
        str(src / "cloudA_step4_socket_client.py"),
        "--qdir", args.qdir,
        "--owner_prf_dir", args.owner_prf_dir,
        "--host", args.host,
        "--port", str(args.port),
    ]

    print("[run-step4-socket] start CloudB server")
    pB = subprocess.Popen(
        cloudB,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    time.sleep(1.0)

    print("[run-step4-socket] start CloudA client")
    pA = subprocess.run(
        cloudA,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("[CloudA output]")
    print(pA.stdout)

    try:
        outB, _ = pB.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        pB.kill()
        outB, _ = pB.communicate()
        print("[run-step4-socket] CloudB timeout, killed")
        print(outB)
        sys.exit(124)

    print("[CloudB output]")
    print(outB)

    if pA.returncode != 0:
        print(f"[run-step4-socket] CloudA failed rc={pA.returncode}")
        sys.exit(pA.returncode)

    if pB.returncode != 0:
        print(f"[run-step4-socket] CloudB failed rc={pB.returncode}")
        sys.exit(pB.returncode)

    print("[run-step4-socket] OK")


if __name__ == "__main__":
    main()