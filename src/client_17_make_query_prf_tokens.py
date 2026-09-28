from pathlib import Path
import json
import numpy as np
import hmac, hashlib
import argparse

KEY_DEFAULT = b"demo-key-change-me"   # 实验用
EPOCH_DEFAULT = "2026-01-26"
DATA_DIR_DEFAULT = r"E:\zyc\project\data\hc"
NODE_NAMES_DEFAULT = "node-names-house-committees.txt"
NODE_LABELS_DEFAULT = "node-labels-house-committees.txt"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", type=str, required=True,
                    help=r'query dir, e.g. E:\zyc\project\out\queries\queryset\q0008')
    ap.add_argument("--data_dir", type=str, default=DATA_DIR_DEFAULT,
                    help="dataset dir containing node-names / node-labels")
    ap.add_argument("--node_names_file", type=str, default=NODE_NAMES_DEFAULT,
                    help="node names filename under data_dir")
    ap.add_argument("--node_labels_file", type=str, default=NODE_LABELS_DEFAULT,
                    help="node labels filename under data_dir")
    ap.add_argument("--epoch", type=str, default=EPOCH_DEFAULT,
                    help="epoch string for PRF input")
    ap.add_argument("--key", type=str, default=None,
                    help="PRF key as UTF-8 string (demo). If omitted, use built-in demo key.")

    # dataset id，默认取 data_dir 的目录名
    ap.add_argument("--dataset_id", type=str, default=None,
                    help="dataset identifier to prepend into PRF input; default = basename(data_dir)")

    # 没有 node names 时，允许直接用原始 vertex ID 作为 sid
    ap.add_argument("--use_vertex_id_as_sid", action="store_true",
                    help="use dataset-native vertex IDs as sid instead of node names")

    # vertex ID 起始基，CH / MA 一般传 1
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
    每一行原样保留为字符串（仅 strip 掉首尾空白），
    不做 int 转换，不排序，不去重，不改内部逗号顺序。
    """
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    args = parse_args()
    q_dir = Path(args.qdir)
    data_dir = Path(args.data_dir)

    node_names = data_dir / args.node_names_file
    node_labels = data_dir / args.node_labels_file

    q_edges = q_dir / "q_edges.json"
    out_dir = q_dir / "online_step4_prf"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not node_labels.exists():
        raise FileNotFoundError(f"Missing {node_labels}")
    if not q_edges.exists():
        raise FileNotFoundError(f"Missing {q_edges}. queryset generator should have written it.")

    key = args.key.encode("utf-8") if args.key is not None else KEY_DEFAULT
    epoch = args.epoch
    dataset_id = args.dataset_id if args.dataset_id is not None else data_dir.name

    # 这里不再假设每行是单个整数；原样保留成字符串
    labels = read_label_lines(node_labels)
    V = len(labels)

    # 统一 sid 规则：要和 owner_01 完全一致
    if args.use_vertex_id_as_sid:
        sids_all = [str(i + args.vertex_id_base) for i in range(V)]
        sid_mode = "vertex_id"
    else:
        if not node_names.exists():
            raise FileNotFoundError(
                f"Missing {node_names}. "
                f"If this dataset has no node names, rerun with --use_vertex_id_as_sid."
            )
        names = [x.strip() for x in node_names.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(names) == len(labels), f"len(names)={len(names)} != len(labels)={len(labels)}"
        sids_all = names
        sid_mode = "node_name"

    qinfo = json.loads(q_edges.read_text(encoding="utf-8"))
    involved = np.array(qinfo["involved_nodes_in_data"], dtype=np.int64)
    U = involved.shape[0]

    # 基本合法性检查
    if involved.ndim != 1:
        raise ValueError(f"involved_nodes_in_data should be 1-D, got shape={involved.shape}")
    if U == 0:
        raise ValueError("involved_nodes_in_data is empty")
    if involved.min() < 0 or involved.max() >= V:
        raise ValueError(
            f"involved node id out of range: min={int(involved.min())}, max={int(involved.max())}, V={V}"
        )

    tok = np.zeros((U, 2), dtype=np.uint64)
    for i, vid in enumerate(involved.tolist()):
        s = f"dataset={dataset_id}|epoch={epoch}|party={labels[vid]}|sid={sids_all[vid]}".encode("utf-8")
        lo, hi = prf128(s, key)
        tok[i, 0] = lo
        tok[i, 1] = hi

    np.save(out_dir / "query_tok_plain.npy", tok)

    # 保存一份参数记录，便于后面查错和复现
    meta = {
        "qdir": str(q_dir),
        "data_dir": str(data_dir),
        "dataset_id": dataset_id,
        "epoch": epoch,
        "party_mode": "raw_label_line",
        "sid_mode": sid_mode,
        "vertex_id_base": int(args.vertex_id_base) if args.use_vertex_id_as_sid else None,
        "U": int(U),
        "token_input_format": "dataset=<dataset_id>|epoch=<epoch>|party=<raw_label_line>|sid=<stable_identifier>"
    }
    (out_dir / "query_tok_params.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )

    print("Saved:", out_dir / "query_tok_plain.npy", "U=", U)
    print("Meta :", out_dir / "query_tok_params.json")
    print("dataset_id =", dataset_id)
    print("sid_mode =", sid_mode)

if __name__ == "__main__":
    main()