# HyPriMatch

This repository contains the research artifact for **HyPriMatch: Privacy-Preserving Pattern Matching over Hypergraphs**. HyPriMatch supports privacy-preserving subhypergraph matching over outsourced, labeled hypergraph data using two non-colluding cloud servers.

The artifact provides the implementation of the data and query preparation procedures, the four protocol stages described in the paper, the end-to-end experiment drivers, and compact result summaries. The implementation simulates the data owner, query client, and two cloud servers on one machine; the communication between the two clouds is implemented with local sockets.

## 1. Repository layout

```text
HyPriMatch/
|-- README.md
|-- src/                          # Protocol and evaluation source code
|   |-- mpc/                      # Beaver multiplication and masked equality
|   `-- socket_sim/               # Socket serialization utilities
|-- out/metrics/                  # HC result summaries
|-- out_ma/metrics/               # MA result summaries
|-- out_wt/metrics/               # WT result summaries
`-- protocol_statistics/          # Aggregated tables used in the paper
```

Large generated files, including dense secret shares, Beaver preprocessing material, temporary arrays, and complete per-query working directories, are not distributed because they can be regenerated from the source code. The original third-party datasets are also not redistributed; their official sources are listed below.

## 2. Correspondence between the paper and the code

| Paper component | Main implementation files |
|---|---|
| Dataset deduplication and incidence-matrix construction | `src/19_dedup_hyperedges_txt.py`, `src/01_build_B.py` |
| Query generation | `src/20_make_queryset_pilot_unicyclic.py` |
| Data structure sharing | `src/owner_03_make_data_structure_shares.py` |
| Data-token generation and sharing | `src/owner_01_make_data_prf_tokens.py`, `src/owner_02_share_data_tokens.py` |
| Query incidence sharing | `src/06_make_dense_shares.py` |
| Secure hyperedge-size and pairwise-intersection pruning (Steps 1--2) | `src/cloudA_step12_onlycloud.py`, `src/cloudB_step12_onlycloud.py`, `src/cloud_step12_beaver_driver.py`, `src/cloudA_step12_socket_client.py`, `src/cloudB_step12_socket_server.py`, `src/run_step12_socket_pair.py` |
| Secure vertex confirmation (Step 3) | `src/client_make_alpha.py`, `src/cloudA_step3_dense.py`, `src/cloudB_step3_dense.py`, `src/client_decrypt_step3_dense.py` |
| Query-token preparation for Step 4 | `src/client_16_make_padded_pairs.py`, `src/client_17_make_query_prf_tokens.py`, `src/client_18_share_query_tokens.py`, `src/client_19_make_step4_preproc.py` |
| Secure token-level consistency verification (Step 4) | `src/cloudA_step4_socket_client.py`, `src/cloudB_step4_socket_server.py`, `src/run_step4_socket_pair.py`, `src/client_step4_prf_filter.py` |
| End-to-end query execution | `src/21_run_queryset_pipeline_socket.py`, `src/24_run_dataset_all_eq.py` |
| Per-query communication accounting | `src/23_collect_query_comm_metrics.py` |
| Aggregate statistics and paper tables | `src/stat_protocol_summary.py` |

The repository also retains several auxiliary implementations and analysis utilities:

- `src/20_make_queryset.py`: a generic query-set generator;
- `src/22_export_runner_summary_csv.py`: an earlier standalone result exporter;
- `src/cloudA_step4_prf_eq_onlypairs.py`, `src/cloudB_step4_prf_eq_onlypairs.py`, and `src/cloudA_step4_finalize.py`: non-socket Step 4 utilities;
- `src/31_analyze_ambiguity_without_plaintext_output.py`: an additional ambiguity-analysis utility.

The socket-based end-to-end experiments reported in the paper are driven by `src/24_run_dataset_all_eq.py` through `src/21_run_queryset_pipeline_socket.py`.

## 3. Datasets

The experiments use three public labeled-hypergraph datasets from the Austin R. Benson data repository.

| ID | Dataset | Official source | Raw hyperedges | Hyperedges after the preprocessing used in this work |
|---|---|---|---:|---:|
| HC | House Committees | <https://www.cs.cornell.edu/~arb/data/house-committees/> | 341 | 336 |
| MA | MathOverflow Answers | <https://www.cs.cornell.edu/~arb/data/mathoverflow-answers/> | 5,446 | 5,445 |
| WT | Walmart Trips | <https://www.cs.cornell.edu/~arb/data/walmart-trips/> | 69,906 | 65,979 |

Download the ZIP archive from each official dataset page and arrange the extracted files as follows:

```text
data/
|-- hc/
|   |-- hyperedges-house-committees.txt
|   |-- node-labels-house-committees.txt
|   |-- node-names-house-committees.txt
|   `-- label-names-house-committees.txt
|-- ma/
|   |-- hyperedges-mathoverflow-answers.txt
|   |-- node-labels-mathoverflow-answers.txt
|   `-- label-names-mathoverflow-answers.txt
`-- wt/
    |-- hyperedges-walmart-trips.txt
    |-- node-labels-walmart-trips.txt
    `-- label-names-walmart-trips.txt
```

The source dataset pages specify the papers and original sources that must be cited when these datasets are used. Dataset rights remain with their respective providers.

### 3.1 Deduplicate hyperedges

HyPriMatch first removes duplicate vertex identifiers within each hyperedge, sorts the identifiers within each hyperedge, and then merges globally duplicated hyperedges while retaining the first occurrence.

```bash
python src/19_dedup_hyperedges_txt.py \
  --infile data/hc/hyperedges-house-committees.txt \
  --outfile data/hc/hyperedges-house-committees-dedup.txt

python src/19_dedup_hyperedges_txt.py \
  --infile data/ma/hyperedges-mathoverflow-answers.txt \
  --outfile data/ma/hyperedges-mathoverflow-answers-dedup.txt

python src/19_dedup_hyperedges_txt.py \
  --infile data/wt/hyperedges-walmart-trips.txt \
  --outfile data/wt/hyperedges-walmart-trips-dedup.txt
```

Each command also generates a `_stats.json` file and an original-to-retained-edge mapping file. The expected final hyperedge counts are 336 for HC, 5,445 for MA, and 65,979 for WT.

### 3.2 Build incidence matrices

```bash
python src/01_build_B.py \
  --hyperedges_path data/hc/hyperedges-house-committees-dedup.txt \
  --node_names_path data/hc/node-names-house-committees.txt \
  --out_dir out \
  --out_npz_name hc_incidence_VxE.npz \
  --out_meta_name hc_meta.json

python src/01_build_B.py \
  --hyperedges_path data/ma/hyperedges-mathoverflow-answers-dedup.txt \
  --node_names_path data/ma/node-labels-mathoverflow-answers.txt \
  --out_dir out_ma \
  --out_npz_name ma_incidence_VxE.npz \
  --out_meta_name ma_meta.json

python src/01_build_B.py \
  --hyperedges_path data/wt/hyperedges-walmart-trips-dedup.txt \
  --node_names_path data/wt/node-labels-walmart-trips.txt \
  --out_dir out_wt \
  --out_npz_name wt_incidence_VxE.npz \
  --out_meta_name wt_meta.json
```

For MA and WT, the node-label file is supplied through the existing `--node_names_path` interface because this preprocessing step uses that file to determine the number of vertices. Stable vertex identifiers are used later when generating PRF tokens for these two datasets.

## 4. Environment

Python 3.10 is recommended. The implementation uses the Python standard library together with:

```text
numpy
scipy
tqdm
```

Install the dependencies with:

```bash
python -m pip install numpy scipy tqdm
```

The scripts contain explicit command-line arguments for input and output paths. Some scripts also retain the authors' machine-specific paths as default examples. Before running an end-to-end experiment, either pass explicit paths or update `PROJECT_ROOT` in `src/24_run_dataset_all_eq.py` to the local repository root.

## 5. Generate query workloads

The paper evaluates query hypergraphs containing 3, 5, 7, and 9 hyperedges. Fifteen queries of each size are evaluated for every dataset. 

The vertex-count ranges used for the four query sizes are:

| Query hyperedges | `u_min` | `u_max` |
|---:|---:|---:|
| 3 | 8 | 200 |
| 5 | 12 | 300 |
| 7 | 16 | 400 |
| 9 | 20 | 500 |

For example, the HC queries with three hyperedges are generated by:

```bash
python src/20_make_queryset_pilot_unicyclic.py \
  --bpath out/hc_incidence_VxE.npz \
  --out_root out/queries/pilot_protocol_hc_eq3 \
  --n_queries 15 \
  --seed 20260201 \
  --eq_target 3 \
  --u_min 8 \
  --u_max 200
```

Repeat the command for query sizes 5, 7, and 9 using the ranges in the table and the corresponding output directory. Replace the incidence-matrix and output paths with `out_ma/ma_incidence_VxE.npz` and `out_wt/wt_incidence_VxE.npz` for MA and WT, respectively. 

The remaining structural thresholds are deterministically derived from `eq_target` by the query-generation script. Generated query metadata records the selected data hyperedges and structural statistics.

## 6. Owner-side preprocessing

The following HC example generates PRF tokens, shares the tokens, and creates additive shares of the data-hypergraph structures:

```bash
python src/owner_01_make_data_prf_tokens.py \
  --data_dir data/hc \
  --node_names_file node-names-house-committees.txt \
  --node_labels_file node-labels-house-committees.txt \
  --out_dir out/hc_owner/prf \
  --dataset_id hc \
  --epoch 2026-01-26

python src/owner_02_share_data_tokens.py \
  --owner_prf_dir out/hc_owner/prf \
  --seed 20260128

python src/owner_03_make_data_structure_shares.py \
  --bpath out/hc_incidence_VxE.npz \
  --out_dir out/data_shares \
  --seed 20260401
```

For MA and WT, use their corresponding data and output directories and add the following options when generating PRF tokens:

```text
--use_vertex_id_as_sid --vertex_id_base 1
```

The fixed seeds and built-in demo PRF key support deterministic experimental simulation. They are not intended as a production key-management mechanism.

## 7. Run the end-to-end experiments

After preparing the datasets, incidence matrices, query workloads, data tokens, and structure shares, execute the 15 queries for every query size as follows:

```bash
python src/24_run_dataset_all_eq.py \
  --dataset hc \
  --eqs 3,5,7,9 \
  --start 1 \
  --end 15 \
  --timeout 0

python src/24_run_dataset_all_eq.py \
  --dataset ma \
  --eqs 3,5,7,9 \
  --start 1 \
  --end 15 \
  --timeout 0

python src/24_run_dataset_all_eq.py \
  --dataset wt \
  --eqs 3,5,7,9 \
  --start 1 \
  --end 15 \
  --timeout 0
```

`--timeout 0` disables the per-step timeout. The experiment driver invokes the socket-based two-cloud pipeline and records per-query latency, correctness, and communication statistics.

## 8. Reported result files

The compact summaries corresponding to the experiments reported in the paper are:

```text
out/metrics/hc_per_query_metrics.csv
out/metrics/hc_summary_metrics.csv
out_ma/metrics/ma_per_query_metrics.csv
out_ma/metrics/ma_summary_metrics.csv
out_wt/metrics/wt_per_query_metrics.csv
out_wt/metrics/wt_summary_metrics.csv
protocol_statistics/per_query_metrics.csv
protocol_statistics/time_table_summary.csv
protocol_statistics/comm_table_summary.csv
protocol_statistics/statistics_full.json
```

The `_pilot_summary.json` files under the query roots record the workload-generation constraints and acceptance statistics. Timing results depend on the local hardware and software environment; correctness and workload structure are the primary deterministic checks.

## 9. Artifact scope

This repository is intended to support inspection of the implementation and its correspondence to the paper. It includes all source modules used by the reported protocol pipeline and compact experimental summaries. The following generated artifacts are intentionally omitted because of their size and because they can be regenerated:

- dense data and query shares;
- Beaver multiplication material;
- PRF token arrays;
- per-query intermediate arrays and socket transcripts;
- complete raw execution logs;
- Python caches and editor-specific files.

## 10. Research-use notice

This code is a research prototype for evaluating the HyPriMatch design. It has not been hardened for production deployment. Users are responsible for complying with the terms and citation requirements of the third-party datasets.
