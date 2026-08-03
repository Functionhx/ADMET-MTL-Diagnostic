"""
B3 R2c P1-5: target-specific label-permuted MTL control.

Reviewer issue: the v1 control permuted a FIXED set of auxiliary endpoints
(heads 1..7) while evaluating all eight heads, so endpoints other than hERG
were evaluated with one genuine auxiliary label present. This version trains
one label-permuted MTL model PER TARGET endpoint: for target t, labels of all
endpoints != t are permuted (frozen per (inst, seed, endpoint) permutations),
the target's labels stay true, and only head t is evaluated. Random protocol,
3 instances x 3 seeds (same scale as the v1 control).

Output: b3_controls_v2_out/control_v2_di.parquet (p_stl / p_mtl / p_mtl_shuff)
plus console summary. Standard MTL and STL columns are reused from v1
(b3_controls_out) where the configuration is identical.
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *
from b3_main_experiment import MLP, fp_array, load_endpoints, global_split
from b3_main_v2 import nll_loss, predict_p

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)
OUT = os.path.join(os.path.dirname(__file__), 'b3_controls_v2_out')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(DATA_DIR)


def train_shuffled(tasks, seed, keep_ep, rng_perm):
    """MTL with auxiliary labels permuted; target endpoint keep_ep stays true."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tasks = [list(t) for t in tasks]
    for h in range(len(tasks)):
        if h != keep_ep:
            Xh, yh = tasks[h]
            tasks[h] = (Xh, yh[rng_perm[h]])
    model = MLP(tasks[0][0].shape[1], n_heads=len(tasks)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    data = [(torch.tensor(X, dtype=torch.float32, device=DEVICE),
             torch.tensor(y, dtype=torch.float32, device=DEVICE)) for X, y in tasks]
    model.train()
    for epoch in range(EPOCHS):
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
    for inst in range(3):
        train_set, test_set = global_split(all_mols, RANDOM_FRAC, seed=SEED_BASE + inst)
        tr_idx, te_idx = {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in train_set])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in test_set])
            if len(tr_idx[ep]) > MAX_TRAIN_PER_ENDPOINT:
                rng = np.random.RandomState(SEED_BASE + inst)
                tr_idx[ep] = rng.choice(tr_idx[ep], MAX_TRAIN_PER_ENDPOINT, replace=False)
        for seed in range(3):
            seed_i = SEED_BASE + seed * 7 + inst
            tasks = [(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].values[tr_idx[ep]]) for ep in eps]
            for h, ep in enumerate(eps):
                rng_perm = {hh: np.random.RandomState(seed_i + 2000 + h * 10 + hh).permutation(
                    len(tasks[hh][1])) for hh in range(len(tasks)) if hh != h}
                shuff = train_shuffled(tasks, seed_i, keep_ep=h, rng_perm=rng_perm)
                X_te = torch.tensor(fps[ep][te_idx[ep]], dtype=torch.float32, device=DEVICE)
                y_te = ep_data[ep]['Y'].values[te_idx[ep]]
                with torch.no_grad():
                    p_shuff = torch.sigmoid(shuff(X_te, head=h)).cpu().numpy()
                rows.append(pd.DataFrame({
                    'inst': inst, 'seed': seed, 'endpoint': ep,
                    'mol': np.array(canon_lists[ep])[te_idx[ep]], 'y': y_te,
                    'p_mtl_shuff': p_shuff}))
            print(f'[{time.time()-t0:.0f}s] inst={inst} seed={seed} done', flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(f'{OUT}/control_v2_di.parquet')
    # 汇总（对照 v1 的 p_stl/p_mtl）
    v1 = pd.read_parquet('b3_controls_out/control_di.parquet')
    v1 = v1[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl', 'p_mtl']]
    m = df.merge(v1, on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')
    for tag, cols in [('MTL', ('p_stl', 'p_mtl')), ('MTL-shuffled(target-specific)', ('p_stl', 'p_mtl_shuff'))]:
        d = m.apply(lambda rr: nll_loss(rr['y'], rr[cols[0]]).mean() -
                    nll_loss(rr['y'], rr[cols[1]]).mean(), axis=1)
        print(f'mean d_{tag} = {d.mean():+.4f}')
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
