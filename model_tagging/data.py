"""User-supplied property graphs; training-only feature and vocabulary fitting."""
import json
from pathlib import Path

import numpy as np


def read_graphs(path, label_names):
    if not label_names or len(set(label_names)) != len(label_names):
        raise ValueError('label_names must be a nonempty unique list.')
    graphs = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = set()
    for graph in graphs:
        if not isinstance(graph['id'], str) or graph['id'] in ids:
            raise ValueError('Graph IDs must be unique strings.')
        ids.add(graph['id'])
        nodes = graph['nodes']
        if not nodes or not set(graph['labels']) <= set(label_names):
            raise ValueError('Empty graph or unknown target label: '+graph['id'])
        for node in nodes:
            if node['role'] not in ('entity', 'property', 'relation') or not isinstance(node['type'], str) or not isinstance(node['value'], str):
                raise ValueError('Invalid node type/value/role.')
            if not isinstance(node['relation_type'], str):
                raise ValueError('Node relation_type must be a string, possibly empty.')
        for edge in graph['edges']:
            if any(type(edge[k]) is not int or not 0 <= edge[k] < len(nodes) for k in ('source', 'target')):
                raise ValueError('Edge endpoint outside node range.')
            if not isinstance(edge['role'], str) or not isinstance(edge['relation_type'], str):
                raise ValueError('Edge role/relation_type must be strings.')
    if len(graphs) < 3:
        raise ValueError('Need at least three graphs for train/validation/test.')
    return graphs


def prepare(graphs, train_indices, config, seed, device):
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from torch_geometric.data import Data

    text = [f"type: {n['type']}; value: {n['value']}" for g in graphs for n in g['nodes']]
    offsets = np.cumsum([0]+[len(g['nodes']) for g in graphs])
    encoder = SentenceTransformer(config['encoder'], device=device)
    embeddings = encoder.encode(text, batch_size=config['embedding_batch_size'], convert_to_numpy=True, show_progress_bar=True)
    training_nodes = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in train_indices])
    dim = config['pca_components']
    if not 0 < dim <= min(len(training_nodes), embeddings.shape[1]):
        raise ValueError('pca_components exceeds the training-node or embedding dimension.')
    pca = PCA(n_components=dim, random_state=seed).fit(embeddings[training_nodes])
    features = pca.transform(embeddings).astype(np.float32)
    training_graphs = [graphs[i] for i in train_indices]
    relation_names = {n['relation_type'] for g in training_graphs for n in g['nodes'] if n['relation_type']}
    relation_names |= {e['relation_type'] for g in training_graphs for e in g['edges'] if e['relation_type']}
    relations = {name: i+1 for i,name in enumerate(sorted(relation_names))}
    roles = {name: i+1 for i,name in enumerate(sorted({e['role'] for g in training_graphs for e in g['edges']}))}
    data = []
    for i,g in enumerate(graphs):
        edges = g['edges']
        item = Data(x=torch.from_numpy(features[offsets[i]:offsets[i+1]]),
                    edge_index=torch.tensor([[e['source'] for e in edges], [e['target'] for e in edges]], dtype=torch.long),
                    node_role=torch.tensor([int(n['role']=='relation') for n in g['nodes']], dtype=torch.long),
                    node_relation_type=torch.tensor([relations.get(n['relation_type'],0) for n in g['nodes']], dtype=torch.long),
                    edge_role=torch.tensor([roles[e['role']] if e['role'] in roles else 0 for e in edges], dtype=torch.long),
                    edge_relation_type=torch.tensor([relations.get(e['relation_type'],0) for e in edges], dtype=torch.long),
                    y=torch.tensor([[float(label in g['labels']) for label in config['label_names']]], dtype=torch.float32))
        data.append(item)
    return data, pca, relations, roles
