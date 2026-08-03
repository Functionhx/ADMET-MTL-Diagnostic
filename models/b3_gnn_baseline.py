"""
B3 GNN baseline (mentor suggestion: cross-architecture robustness).
GIN (graph isomorphism network) on molecular graphs, 8 endpoints,
same leakage-controlled design (global molecule allocation, random + scaffold),
protocol contrast + temperature calibration — shows the observation is not
fingerprint-MLP-specific.
Scale: 2 split instances x 3 seeds (GNN is ~10x slower than MLP).
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)
N_INSTANCES = 5  # matched to MLP (5 instances x 3 seeds)
OUT_DIR = os.path.join(os.path.dirname(__file__), 'b3_main_out')

ATOM_FEAT = 20  # atomic number one-hot up to 20 + degree + aromatic


def mol_to_graph(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    x = []
    for a in m.GetAtoms():
        f = [0.0] * ATOM_FEAT
        z = min(a.GetAtomicNum(), 20) - 1
        f[z] = 1.0
        f.append(float(a.GetDegree()))
        f.append(1.0 if a.GetIsAromatic() else 0.0)
        x.append(f)
    edge_index = [[], []]
    for bond in m.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0].extend([i, j]); edge_index[1].extend([j, i])
    if not edge_index[0]:
        edge_index = [[0], [0]]  # isolated atom fallback
    return Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))


class GIN(nn.Module):
    def __init__(self, d_in, n_heads, hidden=64):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_heads)])

    def forward(self, batch, head=0):
        x, ei, b = batch.x, batch.edge_index, batch.batch
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = F.relu(self.conv3(x, ei))
        h = global_add_pool(x, b)
        return self.heads[head](h).squeeze(-1)


def load_endpoints():
    data = {}
    for ep in ENDPOINTS:
        df = pd.read_csv(os.path.join(DATA_DIR, f'{ep}.csv'))
        df['canon'] = df['Drug'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
                                       if Chem.MolFromSmiles(s) else None)
        data[ep] = df.dropna(subset=['canon']).drop_duplicates(subset=['canon'])
    return data


def global_split(all_mols, frac=0.8, seed=0):
    rng = np.random.RandomState(seed)
    mols = np.array(sorted(all_mols))
    perm = rng.permutation(len(mols))
    cut = int(len(mols) * frac)
    return set(mols[perm[:cut]].tolist()), set(mols[perm[cut:]].tolist())


def scaffold_split(all_mols, frac=0.8, seed=0):
    scaf = {}
    for m in all_mols:
        mol = Chem.MolFromSmiles(m)
        scaf[m] = MurckoScaffoldSmiles(mol=mol) if mol else 'NONE'
    uniq = sorted(set(scaf.values()))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    cut = int(len(uniq) * frac)
    ts = set(uniq[:cut])
    train = {m for m in all_mols if scaf[m] in ts}
    return train, all_mols - train


def train_eval(models_data, seed, stl_ep=None):
    torch.manual_seed(seed); np.random.seed(seed)
    d_in = models_data[0][0][0].x.size(1)
    if stl_ep is not None:
        models_data = [models_data[stl_ep]]
    model = GIN(d_in, len(models_data), hidden=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(80):
        for h, (graphs, ys) in enumerate(models_data):
            n = len(ys)
            for _ in range(8):
                bidx = np.random.randint(0, n, 128)
                batch = Batch.from_data_list([graphs[i] for i in bidx]).to(DEVICE)
                yt = torch.tensor(ys[bidx], dtype=torch.float32, device=DEVICE)
                opt.zero_grad()
                loss = lossf(model(batch, head=h), yt)
                loss.backward(); opt.step()
    model.eval()
    return model


def main():
    t0 = time.time()
    ep_data = load_endpoints()
    canon_lists = {ep: df.canon.tolist() for ep, df in ep_data.items()}
    y_all = {ep: df.Y.to_numpy(float) for ep, df in ep_data.items()}
    # precompute graphs per endpoint (once)
    graphs = {}
    for ep in ENDPOINTS:
        graphs[ep] = [mol_to_graph(s) for s in canon_lists[ep]]
    END_IDS = {ep: i for i, ep in enumerate(ENDPOINTS)}

    results = []
    for protocol, split_fn in [('random', global_split), ('scaffold', scaffold_split)]:
        for inst in range(N_INSTANCES):
            train_set, test_set = split_fn(set.union(*[set(c) for c in canon_lists.values()]),
                                           RANDOM_FRAC, seed=SEED_BASE + inst)
            tr_idx = {ep: [i for i, c in enumerate(canon_lists[ep]) if c in train_set]
                      for ep in ENDPOINTS}
            te_idx = {ep: [i for i, c in enumerate(canon_lists[ep]) if c in test_set]
                      for ep in ENDPOINTS}
            for seed in range(N_SEEDS):
                models_data = [(graphs[ep][i] for i in tr_idx[ep]), None]  # placeholder
                # build per-endpoint graph lists
                mdl = []
                for ep in ENDPOINTS:
                    gs = [graphs[ep][i] for i in tr_idx[ep]]
                    ys = y_all[ep][tr_idx[ep]]
                    mdl.append((gs, ys))
                model_mtl = train_eval(mdl, seed)
                model_stl = {}
                for h, ep in enumerate(ENDPOINTS):
                    model_stl[ep] = train_eval(mdl, seed, stl_ep=h)
                for ep in ENDPOINTS:
                    te = te_idx[ep]
                    if len(te) < MIN_TEST_PER_ENDPOINT:
                        continue
                    batch = Batch.from_data_list([graphs[ep][i] for i in te]).to(DEVICE)
                    yt = y_all[ep][te]
                    eps = 1e-7
                    with torch.no_grad():
                        p_mtl = torch.sigmoid(model_mtl(batch, head=END_IDS[ep])).cpu().numpy()
                        p_stl = torch.sigmoid(model_stl[ep](batch, head=0)).cpu().numpy()
                    nll_mtl = -(yt * np.log(np.clip(p_mtl, eps, 1)) + (1-yt) * np.log(np.clip(1-p_mtl, eps, 1)))
                    nll_stl = -(yt * np.log(np.clip(p_stl, eps, 1)) + (1-yt) * np.log(np.clip(1-p_stl, eps, 1)))
                    d = nll_stl - nll_mtl
                    results.append({'protocol': protocol, 'inst': inst, 'seed': seed,
                                    'endpoint': ep, 'd': d.mean(), 'n': len(te)})
                print(f'[{time.time()-t0:.0f}s] {protocol} inst={inst} seed={seed} done', flush=True)

    df = pd.DataFrame(results)
    df.to_parquet(os.path.join(OUT_DIR, 'b3_gnn_results.parquet'))
    agg = df.groupby('endpoint').apply(
        lambda g: pd.Series({'d_random': g[g.protocol=='random'].d.mean(),
                             'd_scaffold': g[g.protocol=='scaffold'].d.mean()}), include_groups=False)
    agg['Gamma'] = agg.d_scaffold - agg.d_random
    print('\n=== GNN protocol contrast (ΔNLL per endpoint) ===')
    print(agg.round(4).to_string())
    print(f'\nGNN mean d_random={df[df.protocol=="random"].d.mean():+.4f} | '
          f'd_scaffold={df[df.protocol=="scaffold"].d.mean():+.4f} | '
          f'Gamma={agg.Gamma.mean():+.4f}')
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
