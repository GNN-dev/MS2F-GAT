# Zero-shot LLM evaluation

This module implements the manuscript's Raw-XMI / Full-Graph comparison:
the same software models, candidate labels and evaluation subset are used
for each provider and both representations. It uses the project's benchmark
implementation with package-relative paths and a required local configuration.
No API credentials, original data, responses, results or author-selected
experimental configurations are distributed.

## Install and inspect

Use Python 3.10+ and install `requirements.txt` from the repository root.
The LLM module itself requires only NumPy, the OpenAI-compatible Python SDK
and tiktoken; it does not load the graph neural network or sentence encoder.
For LLM-only use, install `numpy openai tiktoken`.

```bash
python -m llm_zero_shot.runner --help
```

Run from the repository root. Relative configuration and data paths resolve
against that root. The first tokenizer use may download a vocabulary into
the ignored `tmp/tiktoken_cache/` directory. Only load graph pickle files
you created or trust.

## Prepare local inputs

The LLM adapter uses **canonical benchmark graphs**, which differ from the
training JSONL interface described in `data/README.md`. It consumes:

| File | Format |
|---|---|
| Labels | UTF-8 CSV with `model_id` and one binary 0/1 column per candidate label; an optional `labels_text` column is ignored. Each row must have at least one positive label. |
| Split | JSON with a nonempty unique `test` ID list; optional `train` and `val`/`validation` lists must not overlap the test set. Supply your own split. |
| Graphs | Pickle containing a list of canonical graph dictionaries, specified below. |
| Raw artifacts | Either a ModelSet ZIP archive or local source files referenced by the graphs. |
| Definitions (optional) | JSON: `{"labels":[{"name":"TAG","definition":"DESCRIPTION"}]}`. Names must match the CSV columns. |

Each graph dictionary requires `model_id`, `labels` (binary vector in CSV
column order), `graph_schema` (string), `graph_schema_version` (integer),
`num_nodes`, `num_edges`, `nodes`, and `edges`. Counts must match array lengths.
For local raw files, include `local_path`. Canonical node records describe
modeling elements (for example `node_id`, `node_type`, `name`, `attributes`);
edge records describe their endpoints and relations. Preserve all graph
nodes and edges. This module serializes supplied graphs; it does not parse
raw XMI into graphs.

Full-Graph requests contain only `directed`, `multigraph`, graph schema
fields, `nodes` and `edges`. Graph IDs, paths and gold labels stay outside
the prompt. Ensure node/edge attributes themselves contain no annotation
metadata. Raw-XMI requests contain the complete decoded source artifact.
The same label catalog is supplied in both conditions, without examples.

## Local configuration interface

Create an ignored local `llm_config.json` object with the following sections.
The tables specify types and constraints, not the manuscript's parameter
values. No configuration file with selected experimental values is bundled.

| Top-level field | Value |
|---|---|
| `protocol_name`, `protocol_version` | Strings identifying your protocol |
| `output_dir` | Local output directory string, preferably under `runs/` |
| `protocol` | Object described below |
| `providers` | Object mapping arbitrary provider names to provider objects |
| `datasets` | Object mapping arbitrary dataset names to dataset objects |

| `protocol` field | Value |
|---|---|
| `tokenizer` | tiktoken encoding name used for common token estimates |
| `context_reserve_tokens` | Nonnegative integer reserved for context overhead |
| `repeats` | Positive integer, number of independent API evaluations |
| `temperature` | Numeric value supported by the chosen endpoints |
| `bootstrap_samples` | Positive integer, graph-level bootstrap draws for reporting |
| `bootstrap_seed` | Nonnegative integer for bootstrap sampling |
| `terminal_failure_categories` (optional) | List of failure categories that halt a provider |
| `consecutive_api_failure_limit` (optional) | Positive integer controlling consecutive-failure shutdown; zero disables that rule |

| Provider field | Value |
|---|---|
| `enabled` | Boolean; at least one provider must be enabled |
| `api_style` | `openai_chat_completions` |
| `base_url` | Your provider's compatible API base URL |
| `model` | Exact model ID returned by the endpoint, including during preflight |
| `key_env` | Name of an environment variable holding the key; never its value |
| `context_window_tokens`, `max_output_tokens` | Positive integers; window must exceed output budget plus context reserve |
| `prompt_token_safety_multiplier` (optional) | Number at least 1, adjusts common tokenizer estimates |
| `max_tokens_parameter` (optional) | Endpoint's output-budget parameter name, e.g. `max_tokens` or `max_completion_tokens` |
| `json_object_mode` (optional) | Boolean controlling the JSON response-format request |
| `omit_temperature` (optional) | Boolean; use for models that do not accept temperature |
| `extra_request` (optional) | Non-secret endpoint-specific generation parameters; do not override model, messages or output budget |
| `tokens_per_minute`, `requests_per_minute` (optional) | Positive integer account limits |
| `request_timeout_seconds` (optional) | Positive numeric timeout |
| `workers` (optional) | Positive integer concurrency; serial execution is the default |

| Dataset field | Value |
|---|---|
| `kind` | `modelset` for an archive, or `v12` for local files (adapter identifiers, not restrictions on dataset names) |
| `split_file`, `labels_file`, `graphs_file` | Paths to the files described above |
| `expected_test_size`, `expected_labels` | Positive integer counts for your inputs |
| `raw_archive`, `raw_archive_prefix` | Required for `modelset`; ZIP path and member prefix |
| `label_definitions_file` (optional) | Path to label definitions |
| `dataset_manifest_file` (optional) | JSON containing `construction_rule.minimum_inclusive_frequency` and `retained_labels` |
| `minimum_label_frequency` | Required when supplying a dataset manifest; must match its frequency field |
| `reference_label_note` (optional) | Local provenance note recorded in the run manifest |

For the archive adapter, model IDs must contain a slash; the suffix after
the first slash is appended to `raw_archive_prefix`. Resolution is exact or
a unique one-segment repair. For the local-file adapter, `local_path` is
relative to the repository root or absolute on your machine.

## Run

Set the environment variables named by each provider's `key_env`. Do not
write keys in configuration files. Prepare and check paired inputs offline:

```bash
python -m llm_zero_shot.runner --config llm_config.json --mode inventory
python -m llm_zero_shot.runner --config llm_config.json --mode prepare
python -m llm_zero_shot.runner --config llm_config.json --mode verify
```

Preparation uses the smallest available input budget across enabled providers,
after output/reserve/safety adjustments. A model exceeding it in either
representation is excluded from both before any API calls; there is no
automatic truncation or chunking. These are common-tokenizer estimates,
so choose the safety multiplier for the endpoints you use.

Use a fresh output directory when changing data, prompts, models or settings.
Then check connectivity, execute the evaluations and report:

```bash
python -m llm_zero_shot.runner --config llm_config.json --mode preflight --execute-paid-api-calls
python -m llm_zero_shot.runner --config llm_config.json --mode run --execute-paid-api-calls
python -m llm_zero_shot.runner --config llm_config.json --mode report
```

`--providers NAME ...` selects providers for preflight/run. The preflight
requires the returned model ID to match the configured ID exactly. Resolve
endpoint aliases before proceeding. After fixing a failed request's cause,
repeat preflight and run with `--retry-failures --execute-paid-api-calls`.
Successful and schema-invalid cached responses are retained. Request hashes
and frozen manifests prevent mismatched results from being reported as complete.

## Outputs and interpretation

All artifacts go to the configured output directory:

- `input_manifest.jsonl`, `run_manifest.json`, `environment.json`: paired
  inputs, hashes, generation settings and package versions;
- `excluded_context_overflow.json`: exclusions before API execution;
- `responses/`: append-only per-provider/dataset/representation/repeat caches;
- `preflight.json`, `preflight_attempts.jsonl`, `run_events.jsonl`: preflight
  results and provider shutdown events;
- `reports/repeat_results.csv`: five metrics for complete evaluations,
  format compliance, invalid-response and API-failure counts;
- `reports/summary.csv`: means, sample SDs and graph-bootstrap intervals;
- `reports/per_label_results.csv`, `failure_summary.csv`, `report_audit.json`:
  per-label scores, failures and completeness checks.

F1 is sample-averaged; precision and recall are micro-averaged. Scores use
the 0–1 scale. A schema-invalid successful response is scored as an empty
tag set. Missing responses and API failures leave the evaluation incomplete;
the report does not score only the successful subset. Run SD is computed
only with multiple complete repeats and left blank with one repeat.
Repeated API calls can yield identical scores; the code does not force
random differences or synthesize results.
