# EvalManager

`EvalManager` evaluates an experiment directory produced by NL-PE retrieval runs. It computes:

- **TREC metrics** (via `pytrec_eval`)
- **Timing analysis** (optional)
- **Per-query and aggregate outputs**

The evaluation operates on an **experiment directory** containing:

```
eval_dir/
├── eval_config.yaml
├── per_query_results/
│   ├── 1/
│   │   ├── trec_results_raw.txt
│   │   └── detailed_results.json
│   ├── 2/
│   │   └── ...
```

---

# Running EvalManager

Basic usage:

```bash
python -m nl_pe.eval_manager -c <eval_dir>
```

Example:

```bash
python -m nl_pe.eval_manager -c trials/ir/scifact/dense
```

---

# Command Line Arguments

| Flag | Description |
|-----|-------------|
| `-c`, `--eval-dir` | Path to experiment directory containing `eval_config.yaml` |
| `-se`, `--skip-existing` | Skip evaluation if all required output files already exist |
| `-t`, `--times` | Enable timing analysis and generate `times.csv` |
| `-strec`, `--skip-trec` | Skip TREC evaluation if `all_queries_trec_eval_results.jsonl` already exists |

Example:

```bash
python -m nl_pe.eval_manager -c trials/ir/scifact/gp_ws -t
```

---

# Output Files

## TREC evaluation

Produces:

```
all_queries_trec_eval_results.jsonl
```

Containing mean and standard deviation of all requested measures.

Example:

```json
{
  "mean_ndcg_cut_10": 0.421,
  "std_dev_ndcg_cut_10": 0.033
}
```

Optional per-query files may also be written:

```
per_query_results/<qid>/trec_eval_results.jsonl
```

---

## Timing analysis

If `-t` is used, the system produces:

```
times.csv
```

Each row corresponds to **one query**.

Example:

```
qid,q_gen,knn,llm_obs,gp_inf,gp_inf_no_IO,tot,llm
1,14.21,0.33,2.15,0.87,0.62,17.56,16.36
2,12.88,0.29,1.94,0.83,0.60,15.94,14.82
```

---

# Timing Columns

Not all columns appear for all experiment types.

| Column | Meaning |
|------|--------|
| `qid` | Query ID |
| `q_gen` | Query generation time |
| `knn` | Dense retrieval time |
| `llm_obs` | Time spent generating LLM observations |
| `gp_inf` | Total GP inference time including IO |
| `gp_inf_no_IO` | GP inference time excluding IO |
| `mmr` | MMR acquisition time |
| `tot` | Total runtime for the query |
| `llm` | Total LLM time (query generation + observations) |

Columns are included **only when relevant to the experiment**.

---

# Experiment Detection

The experiment type is inferred from the directory path.

Example:

```
trials/ir/scifact/gp_ws/llm/gemini-2.5-flash-lite/100/10
```

Parsed as:

```
dataset = scifact
experiment type = gp_ws
```

The experiment type determines which timing components are extracted.

---

# Supported Experiment Types

### Dense Retrieval

```
dense
```

Timing:

```
knn
tot = knn
llm = 0
```

---

### Dense + Query Reformulation

```
dense_q_dec
dense_q_dec_mmr
```

Timing:

```
knn
q_gen
tot = knn + q_gen
llm = q_gen
```

---

### Dense Rerankers

```
dense_oracle
lw
dense_q_dec_oracle
lw_q_dec
```

Timing:

```
knn
llm_obs
[q_gen]
tot = knn + llm_obs + q_gen
llm = llm_obs + q_gen
```

---

### GP Warm Start

```
gp_ws
gp_ws_q_dec
```

Timing components:

```
knn
llm_obs
gp_inf
gp_inf_no_IO
[q_gen]
```

Where:

```
gp_inf =
    final_inf_time
  + final_IO_time
  + model_update_times
```

---

### GP Active Learning

```
gp_al
gp_al_q_dec
```

Timing components:

```
llm_obs
gp_inf
gp_inf_no_IO
[q_gen]
[mmr]
```

Where:

```
gp_inf =
    final_inf_time
  + final_IO_time
  + model_update_times
  + inner_acquisition_times
  + inner_acquisition_IO_times
  + inner_acquisition_sort_times
```

---

# Query Generation Timing

Query generation times are loaded from:

```
data/ir/beir/<dataset>/q_generation/gemini-2.5-flash-lite/5q/q_gen_times_and_parsing.csv
```

Example:

```
qid,parse_success,gen_time
1,1,16.26
3,1,12.42
```

---

# Error Handling

If a value cannot be extracted:

- A warning is logged
- The column value becomes `None`

Example:

```
WARNING Query 12: Missing key 'final_inf_time'
```

The CSV column will still be present.

---

# Skipping Evaluations

### Skip existing outputs

```
-se
```

Skips evaluation if all required output files already exist.

---

### Skip TREC evaluation

```
-strec
```

If:

```
all_queries_trec_eval_results.jsonl
```

already exists, TREC evaluation is skipped.

---

# Typical Workflow

### Run experiment

```
trials/ir/scifact/gp_ws/...
```

### Evaluate retrieval metrics

```bash
python -m nl_pe.eval_manager -c trials/ir/scifact/gp_ws
```

### Evaluate metrics + timing

```bash
python -m nl_pe.eval_manager -c trials/ir/scifact/gp_ws -t
```

### Skip TREC if already evaluated

```bash
python -m nl_pe.eval_manager -c trials/ir/scifact/gp_ws -t -strec
```

---

# Notes

- Timing analysis assumes `detailed_results.json` exists for each query.
- Query generation timing requires the `q_gen_times_and_parsing.csv` file.
- Column presence in `times.csv` depends on experiment type.