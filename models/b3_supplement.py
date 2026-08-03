"""
B3 Supplement (outline review 5/10 fixes):
S1. LEGITIMATE protocol comparator: scaffold-grouped GLOBAL molecule allocation (random vs scaffold as valid protocols) -> protocol x MTL-STL interaction test (C0)
S2. Downsampling experiment: downsample large endpoints (CYP) train sets to small-endpoint sizes (sparsity mechanism, C1)
S3. Within-endpoint continuous novelty model (endpoint controls + size x novelty interaction, C2)
S4. Calibration decomposition: Brier reliability intercept/slope + AUPRC (C5)
S5. Unique molecule counts (hygiene)
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *
from sklearn.metrics import roc_auc_score, average_precision_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)
OUT = os.path.join(os.path.dirname(__file__), 'b3_main_out')


def load_endpoints():
    data = {}
    for ep in ENDPOINTS:
        df = pd.read_csv(os.path.join(DATA_DIR, f'{ep}.csv'))
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


def scaffold_global_split(all_mols, frac=0.8, seed=0):
    """SCAFFOLD-GROUPED global allocation: molecules with same scaffold in one partition."""
    scaf = {}
    for m in all_mols:
        mol = Chem.MolFromSmiles(m)
        scaf[m] = MurckoScaffoldSmiles(mol=mol) if mol else 'NONE'
    uniq = sorted(set(scaf.values()))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    cut = int(len(uniq) * frac)
    train_sc = set(uniq[:cut])
    train_set = {m for m in all_mols if scaf[m] in train_sc}
    test_set = all_mols - train_set
    return train_set, test_set


class MLP(nn.Module):
    def __init__(self, d_in, n_heads=1):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(256, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_heads)])

    def forward(self, x, head=0):
        return self.heads[head](self.backbone(x)).squeeze(-1)


def train_model(tasks, seed, stl_ep=None):
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
def predict(model, X, head=0):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    return torch.sigmoid(model(Xt, head=head)).cpu().numpy()


def run_protocol(ep_data, fps, canon_lists, split_fn, tag, downsample_sizes=None):
    """Run full protocol for a given split function. Returns rows."""
    t0 = time.time()
    all_mols = set()
    for ep in ep_data:
        all_mols |= set(canon_lists[ep])
    rows = []
    for inst in range(min(N_SPLIT_INSTANCES, 3)):  # 3 instances for supplement
        train_set, test_set = split_fn(all_mols, RANDOM_FRAC, seed=SEED_BASE + inst)
        tr_idx, te_idx = {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in train_set])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in test_set])
            if len(tr_idx[ep]) > MAX_TRAIN_PER_ENDPOINT:
                rng = np.random.RandomState(SEED_BASE + inst)
                tr_idx[ep] = rng.choice(tr_idx[ep], MAX_TRAIN_PER_ENDPOINT, replace=False)
            # S2 downsampling: reduce large-endpoint train to match small-endpoint scale
            if downsample_sizes and ep in downsample_sizes:
                rng = np.random.RandomState(SEED_BASE + inst + 7)
                tr_idx[ep] = rng.choice(tr_idx[ep], downsample_sizes[ep], replace=False)
        for seed in range(N_SEEDS):
            tasks = [(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].to_numpy(float)[tr_idx[ep]])
                     for ep in ep_data]
            m_stl = {}
            for h, ep in enumerate(ep_data):
                m_stl[ep] = train_model(tasks, seed, stl_ep=h)
            m_mtl = train_model(tasks, seed)
            for h, ep in enumerate(ep_data):
                y = ep_data[ep]['Y'].to_numpy(float)
                te = te_idx[ep]
                if len(te) < MIN_TEST_PER_ENDPOINT:
                    continue
                p_stl = predict(m_stl[ep], fps[ep][te])
                p_mtl = predict(m_mtl, fps[ep][te], head=h)
                yt = y[te]
                eps = 1e-7
                l_stl = -(yt * np.log(p_stl + eps) + (1 - yt) * np.log(1 - p_stl + eps))
                l_mtl = -(yt * np.log(p_mtl + eps) + (1 - yt) * np.log(1 - p_mtl + eps))
                d = l_stl - l_mtl
                nov = 1 - _jaccard_max(fps[ep][te], fps[ep][tr_idx[ep]])
                for i, ci in enumerate(te):
                    rows.append({'protocol': tag, 'inst': inst, 'seed': seed, 'endpoint': ep,
                                 'mol': canon_lists[ep][ci], 'd': d[i], 'novelty': nov[i],
                                 'y': yt[i], 'p_stl': p_stl[i], 'p_mtl': p_mtl[i]})
        print(f'[{time.time()-t0:.0f}s] {tag} inst={inst} done', flush=True)
    return rows


def _jaccard_max(fps_test, fps_train):
    from sklearn.metrics import pairwise_distances
    sim = 1 - pairwise_distances(fps_test, fps_train, metric='jaccard', n_jobs=8)
    return sim.max(axis=1)


def brier_decomposition(y, p):
    """Brier score + reliability (calibration) and refinement components."""
    brier = np.mean((p - y) ** 2)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, 9)
    rel, res = 0.0, 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        n_b = m.sum()
        rel += n_b / len(y) * (np.mean(p[m]) - np.mean(y[m])) ** 2
        res += n_b / len(y) * (np.mean(y[m]) - np.mean(y)) ** 2
    unc = np.mean(y) * (1 - np.mean(y))
    return brier, rel, res, unc


def main():
    t0 = time.time()
    ep_data = load_endpoints()
    fps = {ep: fp_array(df['canon'].tolist()) for ep, df in ep_data.items()}
    canon_lists = {ep: df['canon'].tolist() for ep, df in ep_data.items()}

    # S1: scaffold protocol (valid protocol comparator)
    rows = run_protocol(ep_data, fps, canon_lists, scaffold_global_split, 'scaffold')
    df_scaf = pd.DataFrame(rows)
    df_scaf.to_parquet(f'{OUT}/b3_scaffold_di.parquet')

    # S2: downsampling large endpoints (CYP2C9/2D6/3A4) to small scale (n=2000)
    ds = {'CYP2C9_Veith': 2000, 'CYP2D6_Veith': 2000, 'CYP3A4_Veith': 2000}
    rows = run_protocol(ep_data, fps, canon_lists, scaffold_global_split, 'scaffold_ds2000',
                        downsample_sizes=ds)
    df_ds = pd.DataFrame(rows)
    df_ds.to_parquet(f'{OUT}/b3_scaffold_ds_di.parquet')

    # Aggregate
    print('\n=== S1: scaffold protocol per-endpoint mean d ===')
    agg = df_scaf.groupby('endpoint').d.agg(['mean', 'count'])
    print(agg.round(4).sort_values('mean'))
    print('\n=== S2: downsampled scaffold per-endpoint mean d (CYP at n=2000) ===')
    agg2 = df_ds.groupby('endpoint').d.agg(['mean', 'count'])
    print(agg2.round(4).sort_values('mean'))

    # S4 calibration decomposition (scaffold data)
    print('\n=== S4: Brier reliability decomposition (scaffold, pooled) ===')
    for ep in ENDPOINTS:
        sub = df_scaf[df_scaf['endpoint'] == ep]
        if len(sub) < 50:
            continue
        y, ps, pm = sub.y.values, sub.p_stl.values, sub.p_mtl.values
        bs, rel, res, unc = brier_decomposition(y, ps)
        bm, relm, resm, uncm = brier_decomposition(y, pm)
        auc_s = roc_auc_score(y, ps)
        auc_m = roc_auc_score(y, pm)
        aupr_s = average_precision_score(y, ps)
        aupr_m = average_precision_score(y, pm)
        print(f'  {ep}: STL brier={bs:.4f} rel={rel:.4f} AUC={auc_s:.4f} AUPRC={aupr_s:.4f} | '
              f'MTL brier={bm:.4f} rel={relm:.4f} AUC={auc_m:.4f} AUPRC={aupr_m:.4f}')

    print(f'\nDone in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
