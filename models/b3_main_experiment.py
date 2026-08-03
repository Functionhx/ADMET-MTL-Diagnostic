"""
B3 R1: Main confirmatory experiment (G0 simulation gate PASSED 2026-08-03).

Frozen design (b3_config.py):
- Global molecule allocation: canonical SMILES -> one partition across ALL endpoints
- 5 split instances x 3 optimization seeds x {STL, hard-sharing MTL}
- Primary estimand: molecule-level log-loss advantage d_i = L_STL,i - L_MTL,i
- Novelty axis: nearest-train Tanimoto (ECFP4), relative to TARGET endpoint's training molecules
- Task sampling: equal expected contribution (primary)
- 9 endpoints, ECFP4 1024-bit, MLP (256,128) shared backbone + heads
- Output: per-molecule d_i table + per-endpoint aggregates (dev evidence excluded)
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)
torch.manual_seed(SEED_BASE)
np.random.seed(SEED_BASE)

DATA = os.path.join(DATA_DIR)
OUT = os.path.join(os.path.dirname(__file__), 'b3_main_out')
os.makedirs(OUT, exist_ok=True)


def load_endpoints():
    data = {}
    for ep in ENDPOINTS:
        df = pd.read_csv(os.path.join(DATA, f'{ep}.csv'))
        df['canon'] = df['Drug'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
                                       if Chem.MolFromSmiles(s) else None)
        df = df.dropna(subset=['canon']).drop_duplicates(subset=['canon'])
        data[ep] = df
    return data


def fp_array(smiles_list):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) if m
                   else AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('C'), 2, 1024))
    return np.array(fps, dtype=np.float32)


def global_split(all_mols, frac=0.8, seed=0):
    """Global molecule allocation: canonical molecules assigned to one partition."""
    rng = np.random.RandomState(seed)
    mols = np.array(sorted(all_mols))
    perm = rng.permutation(len(mols))
    cut = int(len(mols) * frac)
    train_set = set(mols[perm[:cut]].tolist())
    test_set = set(mols[perm[cut:]].tolist())
    return train_set, test_set


def nearest_train_tanimoto(fps_test, fps_train):
    """Per-molecule nearest-train Tanimoto (novelty axis)."""
    from sklearn.metrics import pairwise_distances
    sim = 1 - pairwise_distances(fps_test, fps_train, metric='jaccard', n_jobs=8)
    return sim.max(axis=1)


class MLP(nn.Module):
    def __init__(self, d_in, n_heads=1):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(256, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_heads)])

    def forward(self, x, head=0):
        return self.heads[head](self.backbone(x)).squeeze(-1)


def train_model(tasks, seed, stl_ep=None):
    """tasks: list of (X, y). stl_ep=None -> hard-sharing MTL; else train only that endpoint (STL)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if stl_ep is not None:
        tasks = [tasks[stl_ep]]
    model = MLP(tasks[0][0].shape[1], n_heads=len(tasks)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    data = [(torch.tensor(X, dtype=torch.float32, device=DEVICE),
             torch.tensor(y, dtype=torch.float32, device=DEVICE)) for X, y in tasks]
    model.train()
    UPDATES_PER_TASK = 8  # equal expected contribution: same update count per task per epoch
    for epoch in range(EPOCHS):
        for h, (Xt, yt) in enumerate(data):
            n = len(Xt)
            for _ in range(UPDATES_PER_TASK):
                bidx = torch.randint(0, n, (BATCH,), device=DEVICE)
                opt.zero_grad()
                loss = lossf(model(Xt[bidx], head=h), yt[bidx])
                loss.backward()
                opt.step()
    model.eval()
    return model


@torch.no_grad()
def logloss_pred(model, X, head=0):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    logits = model(Xt, head=head)
    return torch.sigmoid(logits).cpu().numpy()


def main():
    t0 = time.time()
    ep_data = load_endpoints()
    all_mols = set()
    for ep, df in ep_data.items():
        all_mols |= set(df['canon'])
    print(f'[{time.time()-t0:.0f}s] endpoints={len(ep_data)}, unique molecules={len(all_mols)}', flush=True)

    # FPs computed once per endpoint
    fps = {ep: fp_array(df['canon'].tolist()) for ep, df in ep_data.items()}
    canon_lists = {ep: df['canon'].tolist() for ep, df in ep_data.items()}

    all_rows = []
    for inst in range(N_SPLIT_INSTANCES):
        train_set, test_set = global_split(all_mols, RANDOM_FRAC, seed=SEED_BASE + inst)
        # per-endpoint train/test masks from global allocation
        tr_idx, te_idx = {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in train_set])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in test_set])
            if len(tr_idx[ep]) > MAX_TRAIN_PER_ENDPOINT:
                rng = np.random.RandomState(SEED_BASE + inst)
                tr_idx[ep] = rng.choice(tr_idx[ep], MAX_TRAIN_PER_ENDPOINT, replace=False)
        print(f'[{time.time()-t0:.0f}s] inst={inst}: '
              f'train sizes={[len(tr_idx[e]) for e in ep_data][:4]}...', flush=True)

        for seed in range(N_SEEDS):
            tasks = [(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].to_numpy(float)[tr_idx[ep]])
                     for ep in ep_data]
            m_stl_models = {}
            for h, ep in enumerate(ep_data):
                m_stl_models[ep] = train_model(tasks, seed, stl_ep=h)
            m_mtl = train_model(tasks, seed)

            for h, ep in enumerate(ep_data):
                y = ep_data[ep]['Y'].to_numpy(float)
                te = te_idx[ep]
                if len(te) < MIN_TEST_PER_ENDPOINT:
                    continue
                p_stl = logloss_pred(m_stl_models[ep], fps[ep][te], head=0)
                p_mtl = logloss_pred(m_mtl, fps[ep][te], head=h)
                yt = y[te]
                eps = 1e-7
                l_stl = -(yt * np.log(p_stl + eps) + (1 - yt) * np.log(1 - p_stl + eps))
                l_mtl = -(yt * np.log(p_mtl + eps) + (1 - yt) * np.log(1 - p_mtl + eps))
                d = l_stl - l_mtl  # molecule-level log-loss advantage
                nov = nearest_train_tanimoto(fps[ep][te], fps[ep][tr_idx[ep]])
                for i, ci in enumerate(te):
                    all_rows.append({'inst': inst, 'seed': seed, 'endpoint': ep,
                                     'mol': canon_lists[ep][ci], 'd': d[i],
                                     'novelty': nov[i], 'y': yt[i],
                                     'p_stl': p_stl[i], 'p_mtl': p_mtl[i]})
            print(f'[{time.time()-t0:.0f}s] inst={inst} seed={seed} done', flush=True)

    df = pd.DataFrame(all_rows)
    df.to_parquet(os.path.join(OUT, 'b3_main_di.parquet'))
    # aggregate (dev evidence excluded; this is fresh confirmatory data)
    agg = df.groupby(['inst', 'endpoint']).agg(n=('d', 'size'), mean_d=('d', 'mean'),
                                               n_pos=('y', 'sum'), mean_nov=('novelty', 'mean'))
    agg.to_csv(os.path.join(OUT, 'b3_main_agg.csv'))
    print(f'\n=== R1 done in {time.time()-t0:.0f}s ===')
    print(agg.groupby('endpoint').mean_d.agg(['mean', 'std']).round(4))
    print('\nper-instance mean d:')
    print(df.groupby('inst').d.mean().round(4))


if __name__ == '__main__':
    main()
