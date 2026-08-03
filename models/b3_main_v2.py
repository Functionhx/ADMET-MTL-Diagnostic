"""
B3 R1c v2: strict per-model temperature calibration rerun (Reviewer-2 major fix).

Design changes vs v1 (all else frozen from b3_config.py):
- Global molecule allocation into THREE partitions: train 70% / cal 10% / test 20%.
  Random protocol: molecule-level. Scaffold protocol: Bemis-Murcko scaffold-grouped.
- Temperature calibration is now strictly per model: for each (inst, seed, model,
  endpoint), T is fitted by minimizing NLL on THAT model's OWN cal partition
  (disjoint from its train and test) and applied to THAT model's OWN test partition.
- Scaffold protocol upgraded to 5 split instances x 3 seeds (matches random).
- Output: raw + calibrated predictions, molecule-level d and d_cal, novelty
  (nearest-train Tanimoto relative to the 70% train partition), plus
  duplicate/conflict statistics for the data-curation audit.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)
torch.manual_seed(SEED_BASE)
np.random.seed(SEED_BASE)

DATA = os.path.join(DATA_DIR)
OUT = os.path.join(os.path.dirname(__file__), 'b3_v2_out')
os.makedirs(OUT, exist_ok=True)

T_BOUNDS = (0.2, 5.0)


def load_endpoints():
    data = {}
    audit = []
    for ep in ENDPOINTS:
        df = pd.read_csv(os.path.join(DATA, f'{ep}.csv'))
        n_raw = len(df)
        df['canon'] = df['Drug'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
                                       if Chem.MolFromSmiles(s) else None)
        n_parse_fail = df['canon'].isna().sum()
        dup = df[df.duplicated(subset=['canon'], keep=False)]
        n_conflict = 0
        if len(dup) > 0:
            g = dup.groupby('canon')['Y'].nunique()
            n_conflict = int((g > 1).sum())
        df = df.dropna(subset=['canon']).drop_duplicates(subset=['canon'])
        data[ep] = df
        audit.append({'endpoint': ep, 'n_raw': n_raw, 'n_parse_fail': n_parse_fail,
                      'n_conflicting_mols': n_conflict,
                      'n_after_dedup': len(df)})
    audit = pd.DataFrame(audit)
    audit.to_csv(f'{OUT}/data_audit.csv', index=False)
    print(f'data audit: {len(audit)} endpoints, conflicts: {audit.n_conflicting_mols.sum()}',
          flush=True)
    return data


def fp_array(smiles_list):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) if m
                   else AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('C'), 2, 1024))
    return np.array(fps, dtype=np.float32)


def global_split3(all_mols, frac_train=0.7, frac_cal=0.1, seed=0, scaffold=False):
    """Global molecule allocation into train/cal/test (no molecule overlap)."""
    rng = np.random.RandomState(seed)
    if scaffold:
        scaf = {}
        for m in all_mols:
            mol = Chem.MolFromSmiles(m) if m else None
            scaf[m] = MurckoScaffoldSmiles(mol=mol) if mol else 'NONE'
        groups = {}
        for m, sc in scaf.items():
            groups.setdefault(sc, []).append(m)
        g_keys = sorted(groups.keys())
        perm = rng.permutation(len(g_keys))
        n_g = len(g_keys)
        n_train_g = int(n_g * frac_train)
        n_cal_g = int(n_g * frac_cal)
        train_m = [m for k in perm[:n_train_g] for m in groups[g_keys[k]]]
        cal_m = [m for k in perm[n_train_g:n_train_g + n_cal_g] for m in groups[g_keys[k]]]
        test_m = [m for k in perm[n_train_g + n_cal_g:] for m in groups[g_keys[k]]]
        return set(train_m), set(cal_m), set(test_m)
    mols = np.array(sorted(all_mols))
    perm = rng.permutation(len(mols))
    cut1 = int(len(mols) * frac_train)
    cut2 = int(len(mols) * (frac_train + frac_cal))
    return (set(mols[perm[:cut1]].tolist()), set(mols[perm[cut1:cut2]].tolist()),
            set(mols[perm[cut2:]].tolist()))


def nearest_train_tanimoto(fps_test, fps_train):
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
    for epoch in range(EPOCHS):
        for h, (Xt, yt) in enumerate(data):
            n = len(Xt)
            for _ in range(8):  # equal expected contribution (frozen, matches v1)
                bidx = torch.randint(0, n, (BATCH,), device=DEVICE)
                opt.zero_grad()
                loss = lossf(model(Xt[bidx], head=h), yt[bidx])
                loss.backward()
                opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_p(model, X, head=0):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    logits = model(Xt, head=head)
    return torch.sigmoid(logits).cpu().numpy()


def nll_loss(y, p):
    eps = 1e-7
    return -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))


def temp_scale(p, T):
    logit = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    return 1 / (1 + np.exp(-logit / T))


def fit_T(y, p):
    """Per-model scalar temperature by NLL minimization on THAT model's cal partition."""
    r = minimize_scalar(lambda T: nll_loss(y, temp_scale(p, T)).mean(),
                        bounds=T_BOUNDS, method='bounded')
    return r.x


def run_protocol(protocol, ep_data, fps, canon_lists, t0):
    rows = []
    for inst in range(N_SPLIT_INSTANCES):
        train_set, cal_set, test_set = global_split3(
            set().union(*[set(df['canon']) for df in ep_data.values()]),
            seed=SEED_BASE + inst, scaffold=(protocol == 'scaffold'))
        tr_idx, ca_idx, te_idx = {}, {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in train_set])
            ca_idx[ep] = np.array([i for i, c in enumerate(cl) if c in cal_set])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in test_set])
            if len(tr_idx[ep]) > MAX_TRAIN_PER_ENDPOINT:
                rng = np.random.RandomState(SEED_BASE + inst)
                tr_idx[ep] = rng.choice(tr_idx[ep], MAX_TRAIN_PER_ENDPOINT, replace=False)
        print(f'[{time.time()-t0:.0f}s] {protocol} inst={inst}: train cal test sizes '
              f'= {sum(len(v) for v in tr_idx.values())}/{sum(len(v) for v in ca_idx.values())}'
              f'/{sum(len(v) for v in te_idx.values())}', flush=True)
        # novelty depends only on the train partition: compute once per (inst, endpoint)
        nov_by_ep = {ep: nearest_train_tanimoto(fps[ep][te_idx[ep]], fps[ep][tr_idx[ep]])
                     for ep in ep_data}
        for seed in range(N_SEEDS):
            tasks = []
            for ep in ep_data:
                X = fps[ep][tr_idx[ep]]
                y = ep_data[ep]['Y'].values[tr_idx[ep]]
                tasks.append((X, y))
            mtl = train_model(tasks, SEED_BASE + seed * 7 + inst)
            for ep in ep_data:
                X_te = fps[ep][te_idx[ep]]
                y_te = ep_data[ep]['Y'].values[te_idx[ep]]
                X_ca = fps[ep][ca_idx[ep]]
                y_ca = ep_data[ep]['Y'].values[ca_idx[ep]]
                h = list(ep_data.keys()).index(ep)
                stl = train_model(tasks, SEED_BASE + seed * 7 + inst, stl_ep=h)
                p_stl = predict_p(stl, X_te, head=0)
                p_mtl = predict_p(mtl, X_te, head=h)
                # strict per-model calibration: T fit on THIS model's cal partition
                p_stl_ca = predict_p(stl, X_ca, head=0)
                p_mtl_ca = predict_p(mtl, X_ca, head=h)
                T_stl = fit_T(y_ca, p_stl_ca)
                T_mtl = fit_T(y_ca, p_mtl_ca)
                p_stl_cal = temp_scale(p_stl, T_stl)
                p_mtl_cal = temp_scale(p_mtl, T_mtl)
                d = nll_loss(y_te, p_stl).mean(axis=0) - nll_loss(y_te, p_mtl).mean(axis=0)
                d_cal = nll_loss(y_te, p_stl_cal).mean(axis=0) - nll_loss(y_te, p_mtl_cal).mean(axis=0)
                rows.append(pd.DataFrame({
                    'inst': inst, 'seed': seed, 'endpoint': ep,
                    'mol': np.array(canon_lists[ep])[te_idx[ep]], 'y': y_te,
                    'p_stl': p_stl, 'p_mtl': p_mtl,
                    'p_stl_cal': p_stl_cal, 'p_mtl_cal': p_mtl_cal,
                    'd': d, 'd_cal': d_cal, 'novelty': nov_by_ep[ep],
                    'T_stl': T_stl, 'T_mtl': T_mtl}))
            print(f'[{time.time()-t0:.0f}s] {protocol} inst={inst} seed={seed} done', flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(f'{OUT}/b3_v2_{protocol}_di.parquet')
    g = df.groupby('endpoint').d.mean()
    gs = df.groupby('endpoint').d_cal.mean()
    print(f'\n=== v2 {protocol} raw: d_mean={df.d.mean():+.4f} ===')
    print(g.round(4).to_string())
    print(f'=== v2 {protocol} calibrated: d_cal_mean={df.d_cal.mean():+.4f} ===')
    print(gs.round(4).to_string())


def main():
    t0 = time.time()
    ep_data = load_endpoints()
    all_mols = set()
    for ep, df in ep_data.items():
        all_mols |= set(df['canon'])
    print(f'[{time.time()-t0:.0f}s] unique molecules={len(all_mols)}', flush=True)
    fps = {ep: fp_array(df['canon'].tolist()) for ep, df in ep_data.items()}
    canon_lists = {ep: df['canon'].tolist() for ep, df in ep_data.items()}
    if not os.path.exists(f'{OUT}/b3_v2_random_di.parquet'):
        run_protocol('random', ep_data, fps, canon_lists, t0)
    else:
        print('random parquet exists — skipping (resume mode)', flush=True)
    run_protocol('scaffold', ep_data, fps, canon_lists, t0)
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
