# MS²F-GAT

## Multi-Level Structural-Semantic Fusion for Multi-Label Model Tagging in Model Repositories

**Hao Wen · Zhiqiang Li · Min Han · Jianmin Jiang**

This repository accompanies the manuscript **MS²F-GAT: Multi-Level
Structural-Semantic Fusion for Multi-Label Model Tagging in Model
Repositories**. MS²F-GAT assigns multiple tags to software models by
combining the semantic content of modeling elements with their graph
structure.

## Method overview

The implementation organizes model tagging into three components:

1. **Semantic Information Encoding.** Node type/value descriptions are encoded
   with a Sentence Transformer. PCA is fitted on training nodes and applied
   to the remaining graphs.
2. **Structural-Semantic Fusion Encoding.** GATv2 message passing incorporates
   graph topology together with node-role and relation information.
3. **Multi-Level Feature Fusion.** Graph-level semantic features before
   message passing are concatenated with structural-semantic features after
   message passing. An MLP produces a logit for each target tag.

Mean and max pooling are combined in both readout branches. The training
entry point applies sigmoid probabilities and a configurable decision
threshold for multi-label prediction.

## Repository contents

```text
model_tagging/
  model.py       MS²F-GAT encoder, readouts and classifier
  data.py        property-graph loading and semantic features
  train.py       training, validation selection and evaluation
  metrics.py     multi-label evaluation metrics
  __init__.py
llm_zero_shot/
  data.py        paired raw-model and canonical-graph inputs, prompts
  api.py         API requests, JSON validation and rate limiting
  runner.py      preparation, preflight, resumable calls and reporting
  metrics.py     five metrics, per-label scores and bootstrap intervals
  __init__.py
docs/
  llm_zero_shot.md  zero-shot workflow and configuration interface
data/
  README.md      data sources and input specification
requirements.txt
README.md
```

This release provides the core model, an interface for training on
prepared property graphs, and the zero-shot LLM evaluation pipeline for
Raw-XMI and Full-Graph inputs. Raw-model conversion, original benchmark
splits, experiment-specific configurations and supervised baseline runners
are outside its scope. The included training entry point uses random splitting,
unweighted binary cross-entropy and validation sample-F1 for checkpoint
selection; it is not the complete protocol used to produce the manuscript's
experimental tables.

## Installation

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.txt
python -m model_tagging.train --help
python -m llm_zero_shot.runner --help
```

The implementation uses PyTorch, PyTorch Geometric, Sentence Transformers,
NumPy and scikit-learn. Install PyTorch and PyG builds compatible with your
operating system and CUDA version when using a GPU. CPU execution is
supported. The first use of a selected sentence encoder may download its
model files.

## Data preparation

The manuscript considers ModelSet-UML, ModelSet-ECORE and GitUML-Hybrid.
For ModelSet, obtain the data from the [official repository](https://github.com/modelset/modelset-dataset).
Prepare graph and label records according to [data/README.md](data/README.md).
The training interface also accepts other datasets following that format.

Data files, graph caches and the GitUML-Hybrid annotations are not distributed
in this repository. Training requires locally available, labelled property
graphs; the repository does not provide an automatic download or conversion
pipeline for the manuscript datasets.

## Training and evaluation

Create a local JSON configuration and run:

```bash
python -m model_tagging.train \
  --data data/graphs.jsonl \
  --config run_config.json \
  --output runs/experiment
```

For GPU execution, add `--device cuda`. Each run must use a new output
directory. Configuration values are supplied by the user:

| Field | Description |
|---|---|
| `encoder` | Sentence Transformer model name or local model directory |
| `embedding_batch_size` | Positive integer |
| `pca_components` | Positive integer, no larger than the training-node count or encoder dimension |
| `hidden_dim` | Positive integer |
| `num_layers` | Positive integer |
| `heads` | Positive integer |
| `dropout` | Number in [0, 1) |
| `epochs` | Positive integer |
| `batch_size` | Positive integer |
| `learning_rate` | Positive number |
| `weight_decay` | Nonnegative number |
| `threshold` | Number in (0, 1) |
| `train_fraction` | Number in (0, 1) |
| `validation_fraction` | Positive number below 1 minus train_fraction |
| `label_names` | Nonempty list of unique target-label strings |

All listed fields are required. A random seed is generated and recorded by
default; specify `--seed` to reuse a chosen seed. The split is random and does
not guarantee that every rare label appears in each partition. Record the
software environment alongside the run configuration when comparing runs.

## Zero-shot LLM experiments

The zero-shot pipeline evaluates the same models under **Raw-XMI** (complete
source artifact) and **Full-Graph** (complete canonical graph JSON) inputs.
It includes the experimental system/user prompts, stable label IDs, strict
binary JSON decisions, paired context filtering, provider preflight checks,
request-level caches, failure handling and repeated-run summaries.
Endpoints, model IDs and all evaluation settings are supplied locally.

Follow [Zero-shot LLM evaluation](docs/llm_zero_shot.md) to prepare the data
and configuration, then run the offline preparation:

```bash
python -m llm_zero_shot.runner --config llm_config.json --mode prepare
python -m llm_zero_shot.runner --config llm_config.json --mode verify
```

To call the configured APIs and summarize the responses:

```bash
python -m llm_zero_shot.runner --config llm_config.json --mode preflight --execute-paid-api-calls
python -m llm_zero_shot.runner --config llm_config.json --mode run --execute-paid-api-calls
python -m llm_zero_shot.runner --config llm_config.json --mode report
```

Keys are read from environment variables. Responses, metrics and run
manifests remain local. Summary means and sample standard deviations are
computed from completed evaluations; a single evaluation has no run-SD
estimate. The runner does not generate or perturb experimental scores.

## Evaluation metrics

| Metric | Aggregation |
|---|---|
| F1 | F1 computed per graph, then averaged across graphs |
| Recall | Micro-averaged recall over all graph-label decisions |
| Precision | Micro-averaged precision over all graph-label decisions |
| Micro-F1 | F1 computed from pooled true-positive, false-positive and false-negative counts |
| Macro-F1 | F1 computed per label, then averaged across labels |

Scores are reported on the 0–1 scale. Undefined denominators receive zero.
The model selected on the validation set is evaluated on the test set.

## Run outputs

The specified output directory contains:

- `run.json`: configuration, seed, split identifiers, input hash and metrics;
- `model.pt`: trained model parameters and category mappings;
- `pca.joblib`: PCA fitted on training-node embeddings;
- `test_predictions.npz`: test labels and predicted probabilities.

Generated artifacts remain local and are excluded from version control.

## Citation

Please cite the accompanying manuscript when using this implementation:

```bibtex
@unpublished{wen_ms2fgat,
  title  = {{MS$^2$F-GAT}: Multi-Level Structural-Semantic Fusion for
            Multi-Label Model Tagging in Model Repositories},
  author = {Wen, Hao and Li, Zhiqiang and Han, Min and Jiang, Jianmin},
  note   = {Manuscript}
}
```
