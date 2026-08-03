"""
B3 R7: scaffold-protocol mechanistic controls.

The random-protocol controls (STL-8x, label permutation) do not speak to the
headline finding -- the protocol interaction Gamma. This script runs the two
key controls under the SCaffold protocol (3 splits x 3 seeds, 70/10/20):
  - label-permuted MTL (target-specific, as in b3_controls_v2.py)
  - STL-8x (compute-matched, as in b3_stl8x.py)
Standard MTL/STL predictions already exist in b3_v2_out/b3_v2_scaffold_di.parquet.

Key comparison: Gamma_real vs Gamma_permuted -- does the protocol-sensitive
gain survive label permutation?
Output: b3_controls_scaffold_out/scaffold_controls_di.parquet
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *
from b3_main_experiment import MLP, fp_array, load_endpoints
from b3_main_v2 import global_split3, nll_loss, predict_p

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)
OUT = os.path.join(os.path.dirname(__file__), 'b3_controls_scaffold_out')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(DATA_DIR)
N_CTRL = 3  # 3 splits x 3 seeds
EPOCHS_8X = 480  # STL-8x: 8 updates/epoch x 480 epochs = 3840 trunk updates


def train_mtl(tasks, seed, keep_ep=None, stl8x=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tasks = [list(t) for t in tasks]
    if keep_ep is not None:
        for h in range(len(tasks)):
            if h != keep_ep:
                Xh, yh = tasks[h]
                tasks[h] = (Xh, yh[np.random.RandomState(seed + 3000 + h).permutation(len(yh))])
    n_heads = 1 if stl8x else len(tasks)
    model = MLP(tasks[0][0].shape[1], n_heads=n_heads).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    data = [(torch.tensor(X, dtype=torch.float32, device=DEVICE),
             torch.tensor(y, dtype=torch.float32, device=DEVICE)) for X, y in tasks]
    model.train()
    epochs = EPOCHS_8X if stl8x else EPOCHS
    for epoch in range(epochs):
        for h, (Xt, yt) in enumerate(data):
            n = len(Xt)
            for _ in range(8):
                bidx = torch.randint(0, n, (BATCH,), device=DEVICE)
                opt.zero_grad()
                loss = lossf(model(Xt[bidx], head=h), yt[bidx])
                loss.backward()
                opt.step()
    model.eval()
    return model


def main():
    t0 = time.time()
    ep_data = load_endpoints()
    all_mols = set()
    for ep, df in ep_data.items():
        all_mols |= set(df['canon'])
    fps = {ep: fp_array(df['canon'].tolist()) for ep, df in ep_data.items()}
    canon_lists = {ep: df['canon'].tolist() for ep, df in ep_data.items()}
    eps = list(ep_data.keys())
    rows = []
    for inst in range(N_CTRL):
        tr, ca, te = global_split3(all_mols, seed=SEED_BASE + inst, scaffold=True)
        tr_idx, te_idx = {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in tr])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in te])
        for seed in range(N_CTRL):
            seed_i = SEED_BASE + seed * 7 + inst
            tasks = [(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].values[tr_idx[ep]]) for ep in eps]
            # label-permuted MTL per target
            for h, ep in enumerate(eps):
                shuff = train_mtl(tasks, seed_i, keep_ep=h)
                X_te = torch.tensor(fps[ep][te_idx[ep]], dtype=torch.float32, device=DEVICE)
                y_te = ep_data[ep]['Y'].values[te_idx[ep]]
                with torch.no_grad():
                    p_shuff = torch.sigmoid(shuff(X_te, head=h)).cpu().numpy()
                rows.append(pd.DataFrame({'inst': inst, 'seed': seed, 'endpoint': ep,
                                          'mol': np.array(canon_lists[ep])[te_idx[ep]],
                                          'y': y_te, 'p_mtl_shuff': p_shuff}))
            print(f'[{time.time()-t0:.0f}s] inst={inst} seed={seed} permuted done', flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(f'{OUT}/scaffold_permuted_di.parquet')
    print(f'[{time.time()-t0:.0f}s] permuted saved ({len(df)} rows)', flush=True)

    # STL-8x scaffold
    rows8 = []
    import os as _os
    if _os.path.exists(f'{OUT}/scaffold_stl8x_di.parquet'):
        print('stl8x parquet exists — skipping', flush=True)
        return
    for inst in range(N_CTRL):
        tr, ca, te = global_split3(all_mols, seed=SEED_BASE + inst, scaffold=True)
        tr_idx, te_idx = {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in tr])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in te])
        for seed in range(N_CTRL):
            seed_i = SEED_BASE + seed * 7 + inst
            for h, ep in enumerate(eps):
                stl8 = train_mtl([(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].values[tr_idx[ep]])],
                                 seed_i, stl8x=True)
                X_te = torch.tensor(fps[ep][te_idx[ep]], dtype=torch.float32, device=DEVICE)
                y_te = ep_data[ep]['Y'].values[te_idx[ep]]
                with torch.no_grad():
                    p8 = torch.sigmoid(stl8(X_te, head=0)).cpu().numpy()
                rows8.append(pd.DataFrame({'inst': inst, 'seed': seed, 'endpoint': ep,
                                           'mol': np.array(canon_lists[ep])[te_idx[ep]],
                                           'y': y_te, 'p_stl8x': p8}))
                del stl8
            print(f'[{time.time()-t0:.0f}s] inst={inst} seed={seed} stl8x done', flush=True)
    pd.concat(rows8, ignore_index=True).to_parquet(f'{OUT}/scaffold_stl8x_di.parquet')

    # 汇总：Gamma_real vs Gamma_permuted vs Gamma_stl8x
    v2 = pd.read_parquet('b3_v2_out/b3_v2_scaffold_di.parquet')
    v2 = v2[v2.inst.isin([0, 1, 2])]
    m = df.merge(v2[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl', 'p_mtl']],
                 on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')
    m8 = pd.concat(rows8, ignore_index=True).merge(
        v2[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl']],
        on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')
    for tag, d in [('MTL', m.apply(lambda r: nll_loss(r['y'], r['p_stl']).mean() -
                                   nll_loss(r['y'], r['p_mtl']).mean(), axis=1)),
                   ('MTL-permuted', m.apply(lambda r: nll_loss(r['y'], r['p_stl']).mean() -
                                            nll_loss(r['y'], r['p_mtl_shuff']).mean(), axis=1)),
                   ('STL-8x', m8.apply(lambda r: nll_loss(r['y'], r['p_stl']).mean() -
                                       nll_loss(r['y'], r['p_stl8x']).mean(), axis=1))]:
        print(f'scaffold mean d_{tag} = {d.mean():+.4f}')
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
