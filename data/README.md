# Data preparation

MS²F-GAT consumes labelled property graphs. Each model is represented by
semantic nodes, directed edges and a set of target tags. Prepare the graphs
before invoking the training entry point.

## Datasets discussed in the manuscript

| Dataset | Source and availability |
|---|---|
| ModelSet-UML / ModelSet-ECORE | Obtain the source models and metadata from the [ModelSet repository](https://github.com/modelset/modelset-dataset) or its [releases](https://github.com/modelset/modelset-dataset/releases). |
| GitUML-Hybrid | The selected model collection and its annotations are not included in this code release. |

Follow the original providers' data-use terms. ModelSet files require
conversion to the graph schema below. The source data, model selection,
annotations and benchmark splits are not bundled with the implementation.
Locally prepared data can be placed under `data/`; the repository's ignore
rules exclude it from version control.

## Property-graph format

Use UTF-8 JSONL, with one object per graph:

| Field | Required content |
|---|---|
| `id` | Unique graph identifier string |
| `labels` | Target-label strings contained in configuration label_names |
| `nodes` | Nonempty ordered node array |
| `nodes[].type` | Semantic type string |
| `nodes[].value` | Semantic value string |
| `nodes[].role` | entity, property or relation |
| `nodes[].relation_type` | Relation category string, or an empty string |
| `edges` | Directed edge array; may be empty |
| `edges[].source` / `target` | Zero-based integer positions in the node array |
| `edges[].role` | Edge direction/role string |
| `edges[].relation_type` | Relation category string, or an empty string |

Use modeling-element descriptions for node values. Target tags are prediction
labels and must not be inserted into node features as ground-truth metadata.
Relation and property nodes, incidence edges and any reverse edges must be
explicitly present in the supplied graph. The loader uses the supplied
topology without adding graph elements.

## Semantic features and category mappings

Node text is constructed as `type: TYPE; value: VALUE` and encoded with the
configured Sentence Transformer. PCA and the relation/edge-role vocabularies
are fitted using the training partition. Unseen categories use index zero.
Property nodes share the non-relation role embedding in the classifier.
The complete target-label list is supplied through `label_names`.

The original model files, labels, locally generated graph records and fitted
preprocessing artifacts should remain in the user's local data directories.

## Zero-shot LLM inputs

The Raw-XMI / Full-Graph evaluation uses its original canonical benchmark
graph format, a label CSV, a supplied test split and raw model files.
It does not consume the training JSONL above. See
[Zero-shot LLM evaluation](../docs/llm_zero_shot.md) for all fields, archive
and local-file adapters, configuration and commands. Candidate labels are
part of the prompt; gold assignments and graph identity metadata are not.
