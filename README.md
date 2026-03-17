# ReBOL

ReBOL is a research codebase for information retrieval experiments built around modular retrieval pipelines, query reformulation, LLM-based reranking, and Gaussian-process-guided retrieval. The repository is organized so that most experimental variation is expressed through configuration files rather than custom experiment scripts.

## Repository Structure

The main code paths are:

- `src/nl_pe/experiment_manager.py`: main experiment runner
- `src/nl_pe/eval_manager.py`: retrieval evaluation and timing analysis
- `src/nl_pe/search_agent/`: modular retrieval agents, policies, and rerankers
- `src/nl_pe/acitve_learning/`: Gaussian-process active learning code
- `src/nl_pe/embedding/`: embedding and dense retrieval components
- `src/nl_pe/query_gen/`: query reformulation / decomposition generation
- `src/nl_pe/llm/`: prompt-driven LLM interfaces
- `configs/`: reusable configuration templates
- `prompt_templates/`: Jinja prompt templates for scoring, reranking, and generation
- `scripts/`: helpers for running batches of experiments and evaluations
- `trials/`: recommended output location for experiment directories

## Installation

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Several experiments also expect environment variables for LLM providers to be available, typically via a local `.env` file loaded by `python-dotenv`.

## Experiment Lifecycle

Each run is controlled by an **experiment directory** containing a `config.yaml`. The experiment manager reads that config, executes one named routine, and writes outputs back into the same directory.

Typical layout:

```text
trials/<task>/<dataset>/<method>/.../
├── config.yaml
├── experiment.log
└── per_query_results/
```

Single-run execution:

```bash
python src/nl_pe/experiment_manager.py -c <exp_dir> -e <exp_type>
```

Example:

```bash
python src/nl_pe/experiment_manager.py -c trials/ir/scifact/dense -e ir_exp
```

Arguments:

- `-c, --exp-dir`: experiment directory containing `config.yaml`
- `-e, --exp-type`: method of `ExperimentManager` to run
- `-se, --skip-existing`: skip per-query IR runs with existing outputs

Batch execution over many experiment directories:

```bash
python scripts/run_experiment_batch.py -c <batch_dir> -e <exp_type>
```

The batch runner recursively finds all subdirectories containing `config.yaml` and launches the requested experiment in each one.

## Core Experiment Routines

`ExperimentManager` currently exposes the following main routines:

- `index_corpus`: build document embeddings and save a dense index
- `ir_exp`: run query-by-query retrieval experiments
- `gen_qs`: generate reformulated queries
- `gp_inf`: run GP inference speed experiments
- `fit_hyperpriors`: fit hyperprior statistics used by GP experiments
- `tune_indep_gps`: tune one GP per query
- `tune_gp_list`: tune shared GP hyperparameters using an independent model list

For retrieval experiments reported as runs over queries, `ir_exp` is the primary entry point.

## How Retrieval Experiments Are Structured

The retrieval stack is built around three abstractions.

### `GeneralAgent`

`GeneralAgent` is the top-level runtime agent. For each query, it holds a mutable `state` dictionary and repeatedly asks its policy for the next action until termination.

### `PipelinePolicy`

`PipelinePolicy` executes a configured sequence of `(component, method)` steps. This makes the retrieval process declarative: swapping retrieval or reranking behavior usually means editing `agent.policy_steps` rather than changing code.

### Registry-based components

Components are registered centrally in `src/nl_pe/search_agent/registry.py`. Relevant entries include:

- `AgentLogic`
- `HuggingFaceEmbedderSentenceTransformers`
- `GoogleEmbedder`
- `Prompter`
- `GPActiveLearner`
- `TopKPWReranker`
- `LWReranker`

This registry is what allows configs to refer to components by name.

## Baseline Setup

The repository supports several baseline families used in IR comparisons. In practice, a baseline is defined by:

1. the experiment directory name
2. the `config.yaml`
3. the retrieval pipeline specified under `agent.policy_steps`

The evaluation code also infers experiment type from the trial directory structure, so method names should be reflected consistently in folder names.

### Dense retrieval

The simplest baseline is dense retrieval without LLM-based reranking. The typical flow is:

1. embed the query
2. retrieve top-`k` passages from the dense index
3. write the ranked results

Dense retrieval is implemented in the embedder classes, using FAISS or tensor-based exact nearest-neighbor search.

### Query reformulation baselines

Some experiments use query reformulations in addition to the original query. In `ExperimentManager.ir_exp`, reformulations are read from query CSV columns named `q_1`, `q_2`, `...`, while the original query remains the main query string.

This enables methods such as:

- `dense_q_dec`
- `dense_q_dec_mmr`
- `gp_ws_q_dec`
- `gp_al_q_dec`
- `lw_q_dec`
- `dense_q_dec_oracle`

### LLM reranking baselines

LLM reranking is implemented through the `Prompter` scorer and reranker modules. Two common styles are supported:

- **pointwise scoring**: the model assigns a score to each candidate document in a batch
- **listwise reranking**: the model reorders a candidate list directly

Examples referenced by evaluation include:

- `lw`
- `dense_oracle`
- `dense_q_dec_oracle`
- `lw_q_dec`

The `Prompter` class renders a Jinja template, calls the configured LLM, parses the JSON response, and caches results per query and prompt template.

## Gaussian Process / Bayesian Optimization Functionality

The GP functionality is implemented primarily in `src/nl_pe/acitve_learning/active_learners.py` through `GPActiveLearner`.

At a high level, GP-based retrieval treats passage selection as a sequential decision problem:

1. start from an initial observation set, typically containing the query embedding and optionally reformulation embeddings
2. fit a Gaussian process over embedding space
3. use an acquisition function to choose which documents to observe next
4. score selected documents using the configured observation source
5. update the GP
6. after the observation budget is exhausted, rank the full corpus by GP posterior mean

This is Bayesian optimization over the retrieval space: expensive observations are allocated selectively, and the GP is used to generalize those observations to unseen documents.

### Initial observations

The GP pipeline can seed the model with:

- the original query embedding, assigned `gp.query_rel_label`
- reformulation embeddings, if enabled via `gp.use_query_reformulations`
- warm-start document observations from an initial dense ranking, controlled by `gp.warm_start_percent`

### Observation sources

Observed labels come from the configured `observation.class`:

- `gt`: ground-truth qrels supervision via `GTScorer`
- `ce`: cross-encoder scoring
- `llm`: LLM-based scoring via `Prompter`

This allows the same GP machinery to be used with oracle labels, learned labels, or LLM judgments.

### Acquisition functions

The active learner supports multiple acquisition functions and strategies.

Low-level acquisition functions include:

- `ucb_const_beta`
- `ts` (Thompson sampling)
- `greedy`
- `greedy_epsilon`
- `lse_straddle`
- `lse_margin`

Higher-level acquisition strategies include:

- `batch_af`: standard batched acquisition
- `fantasy_af`: sequential fantasy updates within one acquisition round
- `mmr_af`: acquisition with diversity via MMR

### GP fitting and refitting

The GP model is an exact GP with:

- constant mean
- RBF kernel
- Gaussian likelihood

The code supports:

- fixed hyperparameters from config
- optional ARD lengthscales
- periodic hyperparameter refitting during active learning
- separate control over optimizing observation noise and signal variance

Hyperparameter traces such as negative marginal log-likelihood, lengthscale, signal noise, and observation noise are written into query-level result structures.

### Warm-start vs. active-learning variants

The evaluation code distinguishes two major GP retrieval families:

- `gp_ws` / `gp_ws_q_dec`: GP with dense-retrieval warm start
- `gp_al` / `gp_al_q_dec`: GP active learning without dense warm start as the central mechanism

These names matter because evaluation and timing analysis infer method type from the experiment directory path.

## Query Generation

Query generation is handled by `src/nl_pe/query_gen/q_gen.py` through `QueryGenerator`.

The generator:

1. reads source queries from `data.queries_csv_path`
2. renders a prompt template
3. calls an LLM through `Prompter`
4. parses generated outputs into structured columns
5. writes a new CSV plus timing / parsing metadata

Supported writer/prompt patterns include:

- query decomposition (`q_1`, `q_2`, ...)
- elaborated query reformulation
- query-to-document style answer expansion

Generated queries are written to:

```text
<exp_dir>/gen_qs.csv
```

and generation metadata to:

```text
<exp_dir>/q_gen_times_and_parsing.csv
```

These outputs are then consumed by downstream retrieval experiments that expect reformulation columns.

## Indexing and Dense Retrieval

Dense retrieval functionality lives in `src/nl_pe/embedding/embedders.py`.

Supported responsibilities include:

- encoding documents and queries
- writing FAISS exact indexes
- saving raw embedding tensors
- GPU and CPU exact nearest-neighbor retrieval
- MMR-diversified dense retrieval
- multi-query aggregation for reformulation-based retrieval
- optional matryoshka truncation of embedding dimensions

Two embedder backends are currently exposed in the registry:

- `HuggingFaceEmbedderSentenceTransformers`
- `GoogleEmbedder`

For most retrieval experiments, indexing is done once with `index_corpus`, and `ir_exp` then loads the resulting index for query-time retrieval.

## Configuration System

The configuration system is intentionally compositional. A `config.yaml` is typically organized into sections such as:

- `agent`
- `data`
- `embedding`
- `observation`
- `active_learning`
- `gp`
- `optimization`
- `llm`
- `templates`
- `logging`

Two useful reference files are:

- `configs/base_config_dec23.yaml`
- `configs/example_emnlp_config.yaml`

### `agent`

Controls the experiment pipeline.

Important fields include:

- `agent_class`: usually `GeneralAgent`
- `policy`: usually `PipelinePolicy`
- `policy_steps`: ordered list of `(component, method)` calls
- `max_pipeline_iterations`: how many times the pipeline can repeat

Example:

```yaml
agent:
  agent_class: GeneralAgent
  policy: PipelinePolicy
  max_pipeline_iterations: 1
  policy_steps:
    - component: HuggingFaceEmbedderSentenceTransformers
      method: get_query_embedding
    - component: GPActiveLearner
      method: active_learn
```

### `data`

Defines all input and output paths needed by an experiment.

Common fields include:

- query CSVs
- document text CSVs
- dense index paths
- document ID mapping paths
- qrels paths
- cache paths

For retrieval runs, fields such as `q_text_csv`, `index_path`, `doc_ids_path`, and `d_text_csv` are especially important.

### `embedding`

Defines the document/query encoder and dense retrieval behavior.

Common fields include:

- `class`
- `model`
- `normalize`
- `matryoshka_dim`
- `index_method`
- `k`
- `inference_batch_size`
- `query_prompt`
- `doc_prompt`

This section determines both how corpora are indexed and how dense retrieval is executed.

### `observation`

Controls how document labels are obtained during GP or reranking experiments.

Common fields include:

- `class`: `gt`, `ce`, or `llm`
- `normalize_scores`: whether to normalize observed values before GP fitting

### `gp`

Controls GP model structure and GP-specific retrieval behavior.

Important fields include:

- `kernel`
- `lengthscale`
- `signal_noise`
- `observation_noise`
- `query_rel_label`
- `reform_query_rel_label`
- `use_query_reformulations`
- `warm_start_percent`
- `fast_pred`
- `k_final`

Conceptually:

- `lengthscale`, `signal_noise`, and `observation_noise` initialize GP hyperparameters
- `query_rel_label` specifies the supervision value assigned to the query embedding
- `k_final` controls the size of the final ranked list emitted by the GP
- `warm_start_percent` determines how much of an initial dense ranking is labeled before active learning continues

### `active_learning`

Controls the acquisition loop.

Important fields include:

- `n_obs_iterations`
- `acquisition_f`
- `acquisition_strategy`
- `k_acq`
- `ucb_beta_const`
- `mmr_lambda`
- `fantasy_method`
- `epsilon`
- `lse_tau`
- `lse_kappa`

This section determines how new candidate documents are selected for observation.

### `optimization`

Controls GP hyperparameter refitting.

Important fields include:

- `lr`
- `ard`
- `refit_after_obs`
- `k_refit`
- `k_obs_refit`
- `opt_noise`
- `opt_sig_noise`
- `train_iters`

These settings are used by GP tuning routines and, when enabled, by iterative GP refitting during retrieval.

### `llm`

Defines the LLM backend.

Common fields include:

- `model_class`
- `model_name`
- `temperature`
- `num_retries`

This section is used both for LLM-based scoring/reranking and for query generation.

### `templates`

Defines which prompt templates to use.

Common fields include:

- `template_dir`
- `pw_prompt`
- `lw_prompt`
- `template_path`
- helper names for prompt-dictionary construction

The prompt layer is implemented with Jinja2 templates under `prompt_templates/`.

## Example Experiment Patterns

### Dense retrieval baseline

```bash
python src/nl_pe/experiment_manager.py -c trials/ir/scifact/dense -e ir_exp
```

### Query generation

```bash
python src/nl_pe/experiment_manager.py -c trials/ir/scifact/qgen -e gen_qs
```

### Corpus indexing

```bash
python src/nl_pe/experiment_manager.py -c trials/ir/scifact/index -e index_corpus
```

### GP-based retrieval

```bash
python src/nl_pe/experiment_manager.py -c trials/ir/scifact/gp_ws -e ir_exp
```

## Experiment Outputs

For IR runs, outputs are written query-by-query:

```text
<exp_dir>/
├── config.yaml
├── experiment.log
└── per_query_results/
    ├── <qid>/
    │   ├── detailed_results.json
    │   └── trec_results_raw.txt
```

`detailed_results.json` may contain, depending on the method:

- selected documents
- observed scores
- acquisition scores
- GP hyperparameter traces
- timing information
- final ranked passages

`trec_results_raw.txt` contains a TREC-style run for that query.

## Evaluation

Evaluation is handled by `EvalManager`.

Basic usage:

```bash
python -m nl_pe.eval_manager -c <exp_dir>
```

Timing analysis:

```bash
python -m nl_pe.eval_manager -c <exp_dir> -t
```

Useful flags:

- `-se, --skip-existing`: skip evaluation if outputs already exist
- `-t, --times`: produce timing analysis
- `-strec, --skip-trec`: skip TREC evaluation if aggregate results already exist

Outputs include:

- `all_queries_trec_eval_results.jsonl`
- `times.csv` when timing analysis is requested

The evaluator currently distinguishes experiment families such as:

- `dense`
- `dense_q_dec`
- `dense_q_dec_mmr`
- `dense_oracle`
- `lw`
- `dense_q_dec_oracle`
- `lw_q_dec`
- `gp_ws`
- `gp_ws_q_dec`
- `gp_al`
- `gp_al_q_dec`

Timing columns are method-dependent and can include dense retrieval time, query generation time, LLM observation time, GP inference time, and MMR acquisition time. See `eval_README.md` for the exact timing breakdown currently implemented.

## Reproducibility Recommendations

For paper runs, the following conventions are useful:

1. keep exactly one `config.yaml` per experiment directory
2. use directory names that encode dataset and method
3. preserve `experiment.log` and all query-level outputs
4. run evaluation only after all per-query results have been written
5. keep generated queries, indexes, and caches in stable paths referenced from config

## Minimal Config Skeleton

```yaml
agent:
  agent_class: GeneralAgent
  policy: PipelinePolicy
  policy_steps:
    - component: HuggingFaceEmbedderSentenceTransformers
      method: get_query_embedding

data:
  q_text_csv: <path-to-query-csv>
  d_text_csv: <path-to-doc-csv>
  index_path: <path-to-faiss-index>
  doc_ids_path: <path-to-doc-id-pickle>
  qrels_path: <path-to-qrels>

embedding:
  class: HuggingFaceEmbedderSentenceTransformers
  model: sentence-transformers/all-MiniLM-L6-v2
  index_method: embed_all_docs_faiss_exact
  k: 100

observation:
  class: gt

logging:
  level: INFO
```

This skeleton is enough to orient a baseline setup; GP, query-generation, reranking, and LLM experiments extend it by adding the relevant sections described above.
