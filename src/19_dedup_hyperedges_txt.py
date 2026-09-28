from pathlib import Path
import argparse
import json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infile",
        type=str,
        default=r"E:\zyc\project\data\hc\hyperedges-house-committees.txt",
        help="input hyperedges txt, one hyperedge per line, comma-separated vertex ids"
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default=r"E:\zyc\project\data\hc\hyperedges-house-committees-dedup.txt",
        help="output deduplicated hyperedges txt"
    )
    ap.add_argument(
        "--stats",
        type=str,
        default=None,
        help="output stats json path (default: outfile with _stats.json)"
    )
    ap.add_argument(
        "--mapfile",
        type=str,
        default=None,
        help="output mapping json path (default: outfile with _map.json)"
    )
    return ap.parse_args()


def parse_line(line: str, line_no: int):
    """
    把一行解析成整数列表。
    允许空格，分隔符按逗号。
    """
    s = line.strip()
    if not s:
        return []

    parts = [x.strip() for x in s.split(",")]
    vals = []
    for p in parts:
        if p == "":
            continue
        try:
            vals.append(int(p))
        except ValueError:
            raise ValueError(f"Line {line_no}: cannot parse token '{p}' as int")
    return vals


def main():
    args = parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)

    if args.stats is not None:
        stats_path = Path(args.stats)
    else:
        stats_path = outfile.with_name(outfile.stem + "_stats.json")

    if args.mapfile is not None:
        map_path = Path(args.mapfile)
    else:
        map_path = outfile.with_name(outfile.stem + "_map.json")

    if not infile.exists():
        raise FileNotFoundError(f"Missing input file: {infile}")

    lines = infile.read_text(encoding="utf-8").splitlines()

    total_edges_raw = 0
    total_edges_after_internal = 0
    total_edges_final = 0

    num_empty_lines = 0
    num_edges_with_internal_dup = 0
    num_removed_duplicate_edges = 0

    raw_sizes = []
    internal_sizes = []
    final_sizes = []

    # 记录原始边 -> 行内去重排序后的签名
    normalized_edges = []

    for i, line in enumerate(lines, start=1):
        vals = parse_line(line, i)

        if len(vals) == 0:
            num_empty_lines += 1
            continue

        total_edges_raw += 1
        raw_sizes.append(len(vals))

        uniq_sorted = sorted(set(vals))
        if len(uniq_sorted) < len(vals):
            num_edges_with_internal_dup += 1

        internal_sizes.append(len(uniq_sorted))
        normalized_edges.append({
            "orig_edge_index_0based": total_edges_raw - 1,
            "orig_line_no_1based": i,
            "normalized_vertices": uniq_sorted
        })

    total_edges_after_internal = len(normalized_edges)

    # 全局去重
    seen = {}
    dedup_edges = []
    original_to_kept = []

    for item in normalized_edges:
        sig = tuple(item["normalized_vertices"])
        orig_idx = item["orig_edge_index_0based"]

        if sig not in seen:
            new_idx = len(dedup_edges)
            seen[sig] = new_idx
            dedup_edges.append(item["normalized_vertices"])
        else:
            num_removed_duplicate_edges += 1

        kept_idx = seen[sig]
        original_to_kept.append({
            "orig_edge_index_0based": orig_idx,
            "kept_edge_index_0based": kept_idx
        })

    total_edges_final = len(dedup_edges)
    final_sizes = [len(x) for x in dedup_edges]

    # 写 dedup txt
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        for edge in dedup_edges:
            f.write(",".join(str(x) for x in edge) + "\n")

    # 统计
    stats = {
        "input_file": str(infile),
        "output_file": str(outfile),
        "total_lines_in_file": len(lines),
        "num_empty_lines_skipped": num_empty_lines,

        "total_edges_raw": total_edges_raw,
        "total_edges_after_internal_dedup": total_edges_after_internal,
        "total_edges_final_after_global_dedup": total_edges_final,

        "num_edges_with_internal_duplicate_vertices": num_edges_with_internal_dup,
        "num_removed_duplicate_edges": num_removed_duplicate_edges,

        "raw_edge_size_min_avg_max": [
            min(raw_sizes) if raw_sizes else None,
            (sum(raw_sizes) / len(raw_sizes)) if raw_sizes else None,
            max(raw_sizes) if raw_sizes else None,
        ],
        "internal_dedup_edge_size_min_avg_max": [
            min(internal_sizes) if internal_sizes else None,
            (sum(internal_sizes) / len(internal_sizes)) if internal_sizes else None,
            max(internal_sizes) if internal_sizes else None,
        ],
        "final_edge_size_min_avg_max": [
            min(final_sizes) if final_sizes else None,
            (sum(final_sizes) / len(final_sizes)) if final_sizes else None,
            max(final_sizes) if final_sizes else None,
        ],
    }

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 映射
    map_obj = {
        "input_file": str(infile),
        "output_file": str(outfile),
        "description": (
            "Each original hyperedge is internally deduplicated and sorted first; "
            "then globally duplicated hyperedges are merged. "
            "Mapping records original edge index -> kept edge index."
        ),
        "original_edge_count": total_edges_raw,
        "kept_edge_count": total_edges_final,
        "original_to_kept": original_to_kept,
    }
    map_path.write_text(json.dumps(map_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print("Input :", infile)
    print("Output:", outfile)
    print("Stats :", stats_path)
    print("Map   :", map_path)
    print("Raw edges                 :", total_edges_raw)
    print("Edges with internal dups  :", num_edges_with_internal_dup)
    print("Removed duplicate edges   :", num_removed_duplicate_edges)
    print("Final kept edges          :", total_edges_final)


if __name__ == "__main__":
    main()