"""
B3 R5 P0-1: compute-matched STL (STL-8x) baseline.

Reviewer concern: the MTL shared trunk receives 3840 optimizer updates vs 480
for each STL trunk; with auxiliary labels fully permuted the gain persists
(+0.293), so the contrast may reflect optimization budget / input exposure
rather than multi-task transfer.

STL-8x: for each endpoint, train a single-task model using ONLY that endpoint's
true training data, but with 8x the optimizer updates (8 updates/task/epoch x
480 epochs = 3840 trunk updates), identical batch size (512), Adam lr 1e-3,
weight decay 1e-4, with-replacement sampling, and the same step-indexed
schedule as MTL. All other settings frozen.

Interpretation tree (vs standard MTL +0.300 and label-permuted MTL +0.293):
  - STL-8x ~ MTL-shuffled  -> the residual gain is optimization budget
  - MTL-shuffled > STL-8x  -> auxiliary input distribution / noise regularization
  - MTL-real > MTL-shuffled-> genuine label-transfer component

Random protocol, 5 split instances x 3 seeds x 8 endpoints.
Output: b3_stl8x_out/stl8x_di.parquet (p_stl8x per molecule, matching the
v2 random test partitions).
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
OUT = os.path.join(os.path.dirname(__file__), 'b3_stl8x_out')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(DATA_DIR)
UPDATES_PER_EPOCH = 8
EPOCHS_8X = 480  # 8 updates/epoch x 480 epochs = 3840 trunk updates (== MTL trunk)
N_SEEDS_8X = 5   # extra repetitions for stability (5 instances x 5 seeds x 8 endpoints = 200 models)


def train_stl8x(X, y, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X.shape[1], n_heads=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    n = len(Xt)
    model.train()
    for epoch in range(EPOCHS_8X):
        for _ in range(UPDATES_PER_EPOCH):
            bidx = torch.randint(0, n, (BATCH,), device=DEVICE)
            opt.zero_grad()
            loss = lossf(model(Xt[bidx], head=0), yt[bidx])
            loss.backward()
            opt.step()
        if epoch % 120 == 0:
            print(f'[{time.time()-t0:.0f}s] seed={seed} epoch {epoch}/{EPOCHS_8X} '
                  f'loss={loss.item():.4f}', flush=True)
    model.eval()
    return model


def main():
    global t0
    t0 = time.time()
    ep_data = load_endpoints()
    all_mols = set()
    for ep, df in ep_data.items():
        all_mols |= set(df['canon'])
    fps = {ep: fp_array(df['canon'].tolist()) for ep, df in ep_data.items()}
    canon_lists = {ep: df['canon'].tolist() for ep, df in ep_data.items()}
    eps = list(ep_data.keys())
    rows = []
    for inst in range(N_SPLIT_INSTANCES):
        train_set, test_set = global_split(all_mols, RANDOM_FRAC, seed=SEED_BASE + inst)
        tr_idx, te_idx = {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in train_set])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in test_set])
            if len(tr_idx[ep]) > MAX_TRAIN_PER_ENDPOINT:
                rng = np.random.RandomState(SEED_BASE + inst)
                tr_idx[ep] = rng.choice(tr_idx[ep], MAX_TRAIN_PER_ENDPOINT, replace=False)
        for seed in range(N_SEEDS_8X):
            seed_i = SEED_BASE + seed * 7 + inst
            for h, ep in enumerate(eps):
                stl8x = train_stl8x(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].values[tr_idx[ep]], seed_i)
                X_te = torch.tensor(fps[ep][te_idx[ep]], dtype=torch.float32, device=DEVICE)
                y_te = ep_data[ep]['Y'].values[te_idx[ep]]
                with torch.no_grad():
                    p8 = torch.sigmoid(stl8x(X_te, head=0)).cpu().numpy()
                rows.append(pd.DataFrame({
                    'inst': inst, 'seed': seed, 'endpoint': ep,
                    'mol': np.array(canon_lists[ep])[te_idx[ep]], 'y': y_te,
                    'p_stl8x': p8}))
                del stl8x
            print(f'[{time.time()-t0:.0f}s] inst={inst} seed={seed} done', flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(f'{OUT}/stl8x_di.parquet')
    # 汇总：对照 v2 主实验（p_stl/p_mtl）与 controls（p_mtl_shuff）
    v2 = pd.read_parquet('b3_v2_out/b3_v2_random_di.parquet')[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl', 'p_mtl']]
    c2 = pd.read_parquet('b3_controls_v2_out/control_v2_di.parquet')[['inst', 'seed', 'endpoint', 'mol', 'p_mtl_shuff']]
    m = df.merge(v2, on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left').merge(
        c2, on=['inst', 'seed', 'endpoint', 'mol'], how='left')
    for tag, cols in [('STL-8x vs STL', ('p_stl', 'p_stl8x')),
                      ('MTL vs STL-8x', ('p_stl8x', 'p_mtl')),
                      ('MTL-shuffled vs STL-8x', ('p_stl8x', 'p_mtl_shuff')),
                      ('MTL vs STL', ('p_stl', 'p_mtl')),
                      ('MTL-shuffled vs STL', ('p_stl', 'p_mtl_shuff'))]:
        d = m.apply(lambda rr: nll_loss(rr['y'], rr[cols[0]]).mean() -
                    nll_loss(rr['y'], rr[cols[1]]).mean(), axis=1)
        print(f'mean d_{tag} = {d.mean():+.4f}')
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
