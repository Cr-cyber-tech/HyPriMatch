from pathlib import Path
import numpy as np
import json
import hmac, hashlib
import argparse

DATA_DIR_DEFAULT = r"E:\zyc\project\data\hc"
NODE_NAMES_DEFAULT = "node-names-house-committees.txt"
NODE_LABELS_DEFAULT = "node-labels-house-committees.txt"
OUT_DIR_DEFAULT = r"E:\zyc\project\out\hc_owner\prf"

EPOCH_DEFAULT = "2026-01-26"
KEY_DEFAULT = b"demo-key-change-me"   # 实验用：真实论文里不要落盘

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=DATA_DIR_DEFAULT,
                    help="dataset dir containing node names / node labels")
    ap.add_argument("--node_names_file", type=str, default=NODE_NAMES_DEFAULT,
                    help="node names filename under data_dir")
    ap.add_argument("--node_labels_file", type=str, default=NODE_LABELS_DEFAULT,
                    help="node labels filename under data_dir")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT,
                    help="output owner prf dir")
    ap.add_argument("--epoch", type=str, default=EPOCH_DEFAULT,
                    help="epoch string")
    ap.add_argument("--key", type=str, default=None,
                    help="PRF key as UTF-8 string (demo). If omitted, use built-in demo key.")

    # 数据集ID，默认用 data_dir 的最后一级目录名，比如 hc / ch / ma
    ap.add_argument("--dataset_id", type=str, default=None,
                    help="dataset identifier to prepend into PRF input; default = basename(data_dir)")

    # 没有 node names 时，允许直接用原始 vertex ID 当稳定标识 sid
    ap.add_argument("--use_vertex_id_as_sid", action="store_true",
                    help="use dataset-native vertex IDs as sid instead of node names")

    # vertex ID 起始基，CH / MA 这类一般传 1；如果是 0-based 数据集可传 0
    ap.add_argument("--vertex_id_base", type=int, default=1,
                    help="base of dataset-native vertex IDs when using vertex IDs as sid")

    return ap.parse_args()

def prf128(msg: bytes, key: bytes):
    d = hmac.new(key, msg, hashlib.sha256).digest()[:16]
    lo = int.from_bytes(d[:8], "little", signed=False)
    hi = int.from_bytes(d[8:], "little", signed=False)
    return lo, hi

def read_label_lines(path: Path):
    """
    读取 node-labels 文件。
    重要：每一行原样保留为字符串（仅 strip 掉首尾空白），
    不做 int 转换，不排序，不去重，不改内部逗号顺序。
    """
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    node_names = data_dir / args.node_names_file
    node_labels = data_dir / args.node_labels_file

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not node_labels.exists():
        raise FileNotFoundError(f"Missing {node_labels}")

    key = args.key.encode("utf-8") if args.key is not None else KEY_DEFAULT
    epoch = args.epoch

    # dataset_id：如果没显式传，就用 data_dir 的目录名
    dataset_id = args.dataset_id if args.dataset_id is not None else data_dir.name

    # 这里不再假设每行是单个整数；原样保留成字符串
    labels = read_label_lines(node_labels)
    V = len(labels)

    # 统一抽象：sid = stable identifier
    # 有 node names 的数据集：sid = node name
    # 没有 node names 的数据集：sid = dataset-native vertex ID
    if args.use_vertex_id_as_sid:
        sids = [str(i + args.vertex_id_base) for i in range(V)]
        sid_mode = "vertex_id"
    else:
        if not node_names.exists():
            raise FileNotFoundError(
                f"Missing {node_names}. "
                f"If this dataset has no node names, rerun with --use_vertex_id_as_sid."
            )
        names = [x.strip() for x in node_names.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(names) == len(labels), f"len(names)={len(names)} != len(labels)={len(labels)}"
        sids = names
        sid_mode = "node_name"

    tok = np.zeros((V, 2), dtype=np.uint64)

    for i in range(V):
        # PRF 输入：dataset_id + epoch + party(raw label line) + sid
        s = f"dataset={dataset_id}|epoch={epoch}|party={labels[i]}|sid={sids[i]}".encode("utf-8")
        lo, hi = prf128(s, key)
        tok[i, 0] = lo
        tok[i, 1] = hi

    np.save(out_dir / "data_tok_plain.npy", tok)

    (out_dir / "params.json").write_text(
        json.dumps({
            "DATASET_ID": dataset_id,
            "EPOCH": epoch,
            "NOTE": "demo key not stored",
            "data_dir": str(data_dir),
            "node_names_file": args.node_names_file,
            "node_labels_file": args.node_labels_file,
            "party_mode": "raw_label_line",
            "sid_mode": sid_mode,
            "vertex_id_base": int(args.vertex_id_base) if args.use_vertex_id_as_sid else None,
            "token_input_format": "dataset=<dataset_id>|epoch=<epoch>|party=<raw_label_line>|sid=<stable_identifier>"
        }, indent=2),
        encoding="utf-8"
    )

    print("Saved:", out_dir / "data_tok_plain.npy")
    print("V =", V)
    print("dataset_id =", dataset_id)
    print("sid_mode =", sid_mode)

if __name__ == "__main__":
    main()