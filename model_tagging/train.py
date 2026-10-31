"""Train and evaluate the reference model using a user's own configuration."""
import argparse
import hashlib
import json
import random
import secrets
from pathlib import Path

import numpy as np

from .data import prepare, read_graphs
from .metrics import scores

REQUIRED = {'encoder', 'embedding_batch_size', 'pca_components', 'hidden_dim', 'num_layers',
            'heads', 'dropout', 'epochs', 'batch_size', 'learning_rate', 'weight_decay',
            'threshold', 'train_fraction', 'validation_fraction', 'label_names'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Your property-graph JSONL.')
    parser.add_argument('--config', type=Path, required=True, help='Your own JSON configuration; see README.')
    parser.add_argument('--output', type=Path, required=True, help='A new local run directory.')
    parser.add_argument('--seed', type=int, default=None, help='Omit to generate and record a fresh seed.')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if set(cfg) != REQUIRED:
        raise ValueError(f'Configuration keys must be exactly {sorted(REQUIRED)}')
    if not 0 < cfg['train_fraction'] < 1 or not 0 < cfg['validation_fraction'] < 1-cfg['train_fraction']:
        raise ValueError('Invalid split fractions.')
    if not 0 < cfg['threshold'] < 1 or not 0 <= cfg['dropout'] < 1:
        raise ValueError('Invalid threshold/dropout.')
    for key in ('embedding_batch_size','pca_components','hidden_dim','num_layers','heads','epochs','batch_size'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(key+' must be a positive integer.')
    if cfg['learning_rate'] <= 0 or cfg['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings.')
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    if not 0 <= seed < 2**32:
        raise ValueError('Seed must be in [0, 2**32).')
    graphs = read_graphs(args.data, cfg['label_names'])
    order = np.random.default_rng(seed).permutation(len(graphs))
    n_train, n_val = int(len(graphs)*cfg['train_fraction']), int(len(graphs)*cfg['validation_fraction'])
    if min(n_train,n_val,len(graphs)-n_train-n_val) < 1:
        raise ValueError('Split fractions produce an empty partition.')
    partitions = {'train':order[:n_train], 'validation':order[n_train:n_train+n_val], 'test':order[n_train+n_val:]}
    # No heavyweight imports or downloads happen for --help or invalid schemas.
    import joblib
    import torch
    from torch_geometric.loader import DataLoader
    from .model import MS2FGraphClassifier

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    args.output.mkdir(parents=True, exist_ok=False)
    data, pca, relations, roles = prepare(graphs, partitions['train'], cfg, seed, args.device)
    loaders = {name:DataLoader([data[i] for i in indices], batch_size=cfg['batch_size'], shuffle=name=='train') for name,indices in partitions.items()}
    model = MS2FGraphClassifier(variant='full', in_dim=cfg['pca_components'], hidden_dim=cfg['hidden_dim'],
                              out_dim=len(cfg['label_names']), num_layers=cfg['num_layers'], num_relations=len(relations)+1,
                              num_edge_roles=len(roles)+1, dropout=cfg['dropout'], pooling='mean_max', heads=cfg['heads']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    criterion = torch.nn.BCEWithLogitsLoss()

    def predict(loader):
        model.eval()
        truth, prob = [], []
        with torch.no_grad():
            for batch in loader:
                batch=batch.to(args.device)
                truth.append(batch.y.cpu().numpy())
                prob.append(torch.sigmoid(model(batch)).cpu().numpy())
        return np.concatenate(truth), np.concatenate(prob)

    best_score, best_state, best_epoch = -1., None, None
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        for batch in loaders['train']:
            batch=batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(batch),batch.y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            optimizer.step()
        truth, prob = predict(loaders['validation'])
        validation=scores(truth,prob>=cfg['threshold'])
        print(json.dumps({'epoch':epoch,'validation':validation}),flush=True)
        if validation['sample_f1'] > best_score:
            best_score,best_epoch=validation['sample_f1'],epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    truth, prob = predict(loaders['test'])
    result={'seed':seed,'best_epoch':best_epoch,'config':cfg,'test_metrics':scores(truth,prob>=cfg['threshold']),
            'partitions':{k:[graphs[i]['id'] for i in v] for k,v in partitions.items()},
            'input_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
            'torch_version':torch.__version__,'numpy_version':np.__version__}
    (args.output/'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(args.output/'test_predictions.npz',truth=truth,probabilities=prob)
    torch.save({'state_dict':best_state,'config':cfg,'relations':relations,'edge_roles':roles},args.output/'model.pt')
    joblib.dump(pca,args.output/'pca.joblib')
    print(json.dumps(result['test_metrics'],indent=2))


if __name__=='__main__':
    main()
