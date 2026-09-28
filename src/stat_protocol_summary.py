from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# 1. 用户配置
# ============================================================

DATASET_ROOTS = {
    "HC": Path(r"E:\zyc\project\out\queries"),
    "MA": Path(r"E:\zyc\project\out_ma\queries"),
    "WT": Path(r"E:\zyc\project\out_wt\queries"),
}

QUERY_SIZES = [3, 5, 7, 9]
EXPECTED_QUERIES_PER_GROUP = 15

# True：目录缺失、查询不足15个、存在失败查询、字段缺失时立即报错
# False：跳过异常组，并继续统计其他组
STRICT = True

# 输出目录
OUTPUT_DIR = Path(r"E:\zyc\project\protocol_statistics")

# 表格保留的小数位
TIME_DECIMALS = 3
COMM_DECIMALS = 3

# bytes 转换为“MB”的方式。
# 你的原始 JSON 中 online_comm_MB 使用的是 1024^2，因此这里保持一致。
BYTES_PER_MB = 1024 ** 2

# 时间分组。若以后阶段划分发生变化，只需要改这里。
STEP12_TIME_SCRIPTS = [
    "06_make_dense_shares.py",
    "cloudA_step12_onlycloud.py",
    "cloudB_step12_onlycloud.py",
    "run_step12_socket_pair.py",
]

STEP3_TIME_SCRIPTS = [
    "client_make_alpha.py",
    "cloudA_step3_dense.py",
    "cloudB_step3_dense.py",
    "client_decrypt_step3_dense.py",
]

STEP4_TIME_SCRIPTS = [
    "client_16_make_padded_pairs.py",
    "client_17_make_query_prf_tokens.py",
    "client_18_share_query_tokens.py",
    "client_19_make_step4_preproc.py",
    "run_step4_socket_pair.py",
    "client_step4_prf_filter.py",
]

# 这个步骤只是统计通信开销，不计入 Step1/2、Step3 或 Step4 时间：
POSTPROCESS_SCRIPT = "23_collect_query_comm_metrics.py"


# ============================================================
# 2. 基础工具函数
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")

    return data


def find_summary_file(folder: Path) -> tuple[Path, dict[str, Any]]:
    """
    在一个实验目录中寻找包含顶层 results 数组的 summary JSON。

    优先寻找：
        *_summary.json
        *summary*.json

    如果存在多个有效文件，选择修改时间最新的一个，并给出提示。
    """
    candidate_set: set[Path] = set()

    for pattern in ("*_summary.json", "*summary*.json"):
        candidate_set.update(folder.glob(pattern))

    candidates = sorted(
        [p for p in candidate_set if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    valid: list[tuple[Path, dict[str, Any]]] = []

    for path in candidates:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

        if isinstance(data.get("results"), list):
            valid.append((path, data))

    if not valid:
        raise FileNotFoundError(
            f"目录中没有找到包含顶层 results 数组的 summary JSON：\n{folder}"
        )

    if len(valid) > 1:
        print(f"[警告] {folder} 中发现多个有效 summary JSON，使用最新文件：")
        for index, (path, _) in enumerate(valid, start=1):
            prefix = "  ->" if index == 1 else "    "
            print(f"{prefix} {path.name}")

    return valid[0]


def timing_map(result: dict[str, Any]) -> dict[str, float]:
    """
    把 step_timings 转成：
        {脚本文件名: 秒数}
    """
    items = result.get("step_timings")

    if not isinstance(items, list):
        raise KeyError("缺少 step_timings 数组")

    mapping: dict[str, float] = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        step_value = item.get("step")
        seconds_value = item.get("seconds")

        if step_value is None or seconds_value is None:
            continue

        # 即使 step 字段未来变成完整路径，也只保留文件名
        step_name = Path(str(step_value)).name

        if step_name in mapping:
            raise ValueError(f"step_timings 中出现重复步骤：{step_name}")

        mapping[step_name] = float(seconds_value)

    return mapping


def sum_required_steps(
    timings: dict[str, float],
    required_steps: list[str],
    group_name: str,
    query_name: str,
) -> float:
    """检查并累加某个阶段包含的脚本时间。"""
    missing = [step for step in required_steps if step not in timings]

    if missing:
        raise KeyError(
            f"{query_name} 的 {group_name} 缺少步骤：{', '.join(missing)}"
        )

    return sum(timings[step] for step in required_steps)


def require_number(
    mapping: dict[str, Any],
    key: str,
    query_name: str,
) -> float:
    """从字典中读取必需的数值字段。"""
    if key not in mapping:
        raise KeyError(f"{query_name} 缺少字段：{key}")

    value = mapping[key]

    if not isinstance(value, (int, float)):
        raise TypeError(f"{query_name} 的字段 {key} 不是数值：{value!r}")

    return float(value)


def describe(values: list[float]) -> dict[str, float | int]:
    """返回未四舍五入的描述性统计。"""
    if not values:
        raise ValueError("无法统计空列表")

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """
    用 utf-8-sig 输出 CSV，便于 Windows Excel/WPS 直接打开。
    """
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 3. 提取单个查询的数据
# ============================================================

def extract_query_record(
    result: dict[str, Any],
    dataset: str,
    query_size: int,
    summary_file: Path,
) -> dict[str, Any]:
    """提取一个 q0001/q0002... 查询的时间与通信数据。"""
    query_name = str(result.get("q", "unknown"))
    status = result.get("status")

    if status != "ok":
        raise RuntimeError(
            f"{dataset}-q{query_size}-{query_name} 的状态不是 ok：{status!r}"
        )

    timings = timing_map(result)

    sec_step12 = sum_required_steps(
        timings,
        STEP12_TIME_SCRIPTS,
        "secStep1/2",
        query_name,
    )
    sec_step3 = sum_required_steps(
        timings,
        STEP3_TIME_SCRIPTS,
        "secStep3",
        query_name,
    )
    sec_step4 = sum_required_steps(
        timings,
        STEP4_TIME_SCRIPTS,
        "secStep4",
        query_name,
    )

    grouped_total = sec_step12 + sec_step3 + sec_step4

    recorded_seconds_raw = result.get("seconds")
    recorded_seconds = (
        float(recorded_seconds_raw)
        if isinstance(recorded_seconds_raw, (int, float))
        else None
    )

    total_difference = (
        grouped_total - recorded_seconds
        if recorded_seconds is not None
        else None
    )

    # 验证三个阶段之和与 summary 中的 seconds 一致
    if total_difference is not None and abs(total_difference) > 1e-6:
        raise ValueError(
            f"{dataset}-q{query_size}-{query_name} 的阶段时间之和 "
            f"{grouped_total:.9f}s 与 recorded seconds "
            f"{recorded_seconds:.9f}s 不一致，差值={total_difference:.9f}s"
        )

    comm = result.get("comm_metrics")
    if not isinstance(comm, dict):
        raise KeyError(f"{query_name} 缺少 comm_metrics 对象")

    step12_online_bytes = require_number(
        comm, "step12_online_comm_bytes", query_name
    )
    step12_offline_bytes = require_number(
        comm, "step12_offline_comm_bytes", query_name
    )
    step4_online_bytes = require_number(
        comm, "step4_online_comm_bytes", query_name
    )
    step4_offline_bytes = require_number(
        comm, "step4_offline_comm_bytes", query_name
    )

    return {
        "dataset": dataset,
        "query_size": query_size,
        "query": query_name,
        "status": status,
        "summary_file": str(summary_file),

        "secStep1_2_s": sec_step12,
        "secStep3_s": sec_step3,
        "secStep4_s": sec_step4,
        "grouped_total_s": grouped_total,
        "recorded_seconds_s": recorded_seconds,
        "time_difference_s": total_difference,

        "secStep1_2_online_bytes": int(step12_online_bytes),
        "secStep1_2_offline_bytes": int(step12_offline_bytes),
        "secStep4_online_bytes": int(step4_online_bytes),
        "secStep4_offline_bytes": int(step4_offline_bytes),

        "secStep1_2_online_MB": step12_online_bytes / BYTES_PER_MB,
        "secStep1_2_offline_MB": step12_offline_bytes / BYTES_PER_MB,
        "secStep4_online_MB": step4_online_bytes / BYTES_PER_MB,
        "secStep4_offline_MB": step4_offline_bytes / BYTES_PER_MB,
    }


# ============================================================
# 4. 读取全部 12 组实验
# ============================================================

def collect_all_records() -> tuple[list[dict[str, Any]], list[str]]:
    all_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for query_size in QUERY_SIZES:
        for dataset, root in DATASET_ROOTS.items():
            folder = root / f"pilot_protocol_{dataset.lower()}_eq{query_size}"

            print("=" * 78)
            print(f"[读取] 数据集={dataset}, 查询规模=q{query_size}")
            print(f"[目录] {folder}")

            try:
                if not folder.is_dir():
                    raise FileNotFoundError(f"实验目录不存在：{folder}")

                summary_file, data = find_summary_file(folder)
                print(f"[文件] {summary_file.name}")

                results = data.get("results")
                if not isinstance(results, list):
                    raise KeyError("summary JSON 缺少 results 数组")

                total_count = len(results)
                ok_results = [
                    item
                    for item in results
                    if isinstance(item, dict) and item.get("status") == "ok"
                ]
                failed_results = [
                    item
                    for item in results
                    if not isinstance(item, dict) or item.get("status") != "ok"
                ]

                print(
                    f"[数量] total={total_count}, "
                    f"ok={len(ok_results)}, fail={len(failed_results)}"
                )

                if total_count != EXPECTED_QUERIES_PER_GROUP:
                    raise ValueError(
                        f"查询总数应为 {EXPECTED_QUERIES_PER_GROUP}，"
                        f"实际为 {total_count}"
                    )

                if failed_results:
                    failed_names = [
                        str(item.get("q", "unknown"))
                        if isinstance(item, dict)
                        else "invalid_item"
                        for item in failed_results
                    ]
                    raise RuntimeError(
                        "存在失败查询：" + ", ".join(failed_names)
                    )

                group_records = [
                    extract_query_record(
                        result=item,
                        dataset=dataset,
                        query_size=query_size,
                        summary_file=summary_file,
                    )
                    for item in ok_results
                ]

                if len(group_records) != EXPECTED_QUERIES_PER_GROUP:
                    raise ValueError(
                        f"有效查询数应为 {EXPECTED_QUERIES_PER_GROUP}，"
                        f"实际为 {len(group_records)}"
                    )

                all_records.extend(group_records)
                print("[完成] 该组数据读取与验证成功")

            except Exception as exc:
                message = f"{dataset}-q{query_size}: {exc}"
                errors.append(message)
                print(f"[错误] {message}")

                if STRICT:
                    raise

    return all_records, errors


# ============================================================
# 5. 聚合并输出表格数据
# ============================================================

def aggregate_records(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    返回：
      time_rows：图一表格数据
      comm_rows：图二表格数据
      full_stats_rows：未四舍五入的完整统计
    """
    time_rows: list[dict[str, Any]] = []
    comm_rows: list[dict[str, Any]] = []
    full_stats_rows: list[dict[str, Any]] = []

    time_metrics = [
        "secStep1_2_s",
        "secStep3_s",
        "secStep4_s",
    ]

    comm_metrics = [
        "secStep1_2_online_MB",
        "secStep1_2_offline_MB",
        "secStep4_online_MB",
        "secStep4_offline_MB",
    ]

    for query_size in QUERY_SIZES:
        for dataset in DATASET_ROOTS:
            group = [
                row
                for row in records
                if row["dataset"] == dataset
                and row["query_size"] == query_size
            ]

            if not group:
                continue

            metric_stats: dict[str, dict[str, float | int]] = {}

            for metric in time_metrics + comm_metrics:
                metric_stats[metric] = describe(
                    [float(row[metric]) for row in group]
                )

            time_rows.append({
                "query_size": query_size,
                "dataset": dataset,
                "n": len(group),
                "secStep1_2_s": round(
                    float(metric_stats["secStep1_2_s"]["mean"]),
                    TIME_DECIMALS,
                ),
                "secStep3_s": round(
                    float(metric_stats["secStep3_s"]["mean"]),
                    TIME_DECIMALS,
                ),
                "secStep4_s": round(
                    float(metric_stats["secStep4_s"]["mean"]),
                    TIME_DECIMALS,
                ),
            })

            comm_rows.append({
                "query_size": query_size,
                "dataset": dataset,
                "n": len(group),
                "secStep1_2_online_MB": round(
                    float(metric_stats["secStep1_2_online_MB"]["mean"]),
                    COMM_DECIMALS,
                ),
                "secStep1_2_offline_MB": round(
                    float(metric_stats["secStep1_2_offline_MB"]["mean"]),
                    COMM_DECIMALS,
                ),
                "secStep4_online_MB": round(
                    float(metric_stats["secStep4_online_MB"]["mean"]),
                    COMM_DECIMALS,
                ),
                "secStep4_offline_MB": round(
                    float(metric_stats["secStep4_offline_MB"]["mean"]),
                    COMM_DECIMALS,
                ),
            })

            full_stats_rows.append({
                "query_size": query_size,
                "dataset": dataset,
                "n": len(group),
                "metrics": metric_stats,
            })

    return time_rows, comm_rows, full_stats_rows


def write_latex_rows(
    path: Path,
    time_rows: list[dict[str, Any]],
    comm_rows: list[dict[str, Any]],
) -> None:
    """输出可直接粘贴到 LaTeX 表格中的数据行。"""
    lines: list[str] = []

    lines.append("% ==================================================")
    lines.append("% Table 1: Query time")
    lines.append("% q-size & Dataset & secStep1/2 & secStep3 & secStep4")
    lines.append("% ==================================================")

    for index, row in enumerate(time_rows):
        lines.append(
            f"$q_{{{row['query_size']}}}$ & "
            f"{row['dataset']} & "
            f"{row['secStep1_2_s']:.{TIME_DECIMALS}f} & "
            f"{row['secStep3_s']:.{TIME_DECIMALS}f} & "
            f"{row['secStep4_s']:.{TIME_DECIMALS}f} \\\\"
        )

        next_is_new_size = (
            index == len(time_rows) - 1
            or time_rows[index + 1]["query_size"] != row["query_size"]
        )
        if next_is_new_size and index != len(time_rows) - 1:
            lines.append(r"\midrule")

    lines.append("")
    lines.append("% ==================================================")
    lines.append("% Table 2: Communication cost")
    lines.append(
        "% q-size & Dataset & Step1/2 Online & Step1/2 Offline "
        "& Step4 Online & Step4 Offline"
    )
    lines.append("% ==================================================")

    for index, row in enumerate(comm_rows):
        lines.append(
            f"$q_{{{row['query_size']}}}$ & "
            f"{row['dataset']} & "
            f"{row['secStep1_2_online_MB']:.{COMM_DECIMALS}f} & "
            f"{row['secStep1_2_offline_MB']:.{COMM_DECIMALS}f} & "
            f"{row['secStep4_online_MB']:.{COMM_DECIMALS}f} & "
            f"{row['secStep4_offline_MB']:.{COMM_DECIMALS}f} \\\\"
        )

        next_is_new_size = (
            index == len(comm_rows) - 1
            or comm_rows[index + 1]["query_size"] != row["query_size"]
        )
        if next_is_new_size and index != len(comm_rows) - 1:
            lines.append(r"\midrule")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_time_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("图一：平均查询时间（单位：s）")
    print("=" * 78)
    print(
        f"{'Size':<8}"
        f"{'Dataset':<10}"
        f"{'secStep1/2':>16}"
        f"{'secStep3':>14}"
        f"{'secStep4':>14}"
        f"{'n':>6}"
    )

    for row in rows:
        print(
            f"{('q' + str(row['query_size'])):<8}"
            f"{row['dataset']:<10}"
            f"{row['secStep1_2_s']:>16.{TIME_DECIMALS}f}"
            f"{row['secStep3_s']:>14.{TIME_DECIMALS}f}"
            f"{row['secStep4_s']:>14.{TIME_DECIMALS}f}"
            f"{row['n']:>6}"
        )


def print_comm_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 96)
    print("图二：平均通信开销（单位：MB，按 bytes / 1024^2）")
    print("=" * 96)
    print(
        f"{'Size':<8}"
        f"{'Dataset':<10}"
        f"{'S1/2 Online':>16}"
        f"{'S1/2 Offline':>16}"
        f"{'S4 Online':>16}"
        f"{'S4 Offline':>16}"
        f"{'n':>6}"
    )

    for row in rows:
        print(
            f"{('q' + str(row['query_size'])):<8}"
            f"{row['dataset']:<10}"
            f"{row['secStep1_2_online_MB']:>16.{COMM_DECIMALS}f}"
            f"{row['secStep1_2_offline_MB']:>16.{COMM_DECIMALS}f}"
            f"{row['secStep4_online_MB']:>16.{COMM_DECIMALS}f}"
            f"{row['secStep4_offline_MB']:>16.{COMM_DECIMALS}f}"
            f"{row['n']:>6}"
        )


# ============================================================
# 6. 主函数
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records, errors = collect_all_records()

    if not records:
        raise RuntimeError("没有提取到任何有效查询记录")

    time_rows, comm_rows, full_stats_rows = aggregate_records(records)

    # 逐查询明细：保留原始精度
    per_query_fields = [
        "dataset",
        "query_size",
        "query",
        "status",
        "summary_file",

        "secStep1_2_s",
        "secStep3_s",
        "secStep4_s",
        "grouped_total_s",
        "recorded_seconds_s",
        "time_difference_s",

        "secStep1_2_online_bytes",
        "secStep1_2_offline_bytes",
        "secStep4_online_bytes",
        "secStep4_offline_bytes",

        "secStep1_2_online_MB",
        "secStep1_2_offline_MB",
        "secStep4_online_MB",
        "secStep4_offline_MB",
    ]

    write_csv(
        OUTPUT_DIR / "per_query_metrics.csv",
        records,
        per_query_fields,
    )

    # 图一表格
    write_csv(
        OUTPUT_DIR / "time_table_summary.csv",
        time_rows,
        [
            "query_size",
            "dataset",
            "n",
            "secStep1_2_s",
            "secStep3_s",
            "secStep4_s",
        ],
    )

    # 图二表格
    write_csv(
        OUTPUT_DIR / "comm_table_summary.csv",
        comm_rows,
        [
            "query_size",
            "dataset",
            "n",
            "secStep1_2_online_MB",
            "secStep1_2_offline_MB",
            "secStep4_online_MB",
            "secStep4_offline_MB",
        ],
    )

    # 未四舍五入的 mean/std/min/max
    full_output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expected_queries_per_group": EXPECTED_QUERIES_PER_GROUP,
        "bytes_per_MB": BYTES_PER_MB,
        "time_groups": {
            "secStep1_2_s": STEP12_TIME_SCRIPTS,
            "secStep3_s": STEP3_TIME_SCRIPTS,
            "secStep4_s": STEP4_TIME_SCRIPTS,
            "excluded_postprocess_script": POSTPROCESS_SCRIPT,
        },
        "groups": full_stats_rows,
        "errors": errors,
    }

    with (OUTPUT_DIR / "statistics_full.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)

    write_latex_rows(
        OUTPUT_DIR / "latex_table_rows.txt",
        time_rows,
        comm_rows,
    )

    print_time_table(time_rows)
    print_comm_table(comm_rows)

    print("\n" + "=" * 78)
    print("统计完成，输出目录：")
    print(OUTPUT_DIR)
    print("\n生成文件：")
    print("  1. per_query_metrics.csv")
    print("  2. time_table_summary.csv")
    print("  3. comm_table_summary.csv")
    print("  4. statistics_full.json")
    print("  5. latex_table_rows.txt")

    if errors:
        print("\n以下实验组被跳过：")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()