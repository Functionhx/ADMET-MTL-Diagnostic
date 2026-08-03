"""
B3 R8: validation-selected checkpoint sensitivity (random protocol, 3x3).

Reviewer concern: raw log-loss differences may reflect fixed final checkpoints
(overconfident STL at fixed epochs) rather than an MTL evaluation phenomenon.

Design: the 10% calibration partition is split into 5% validation + 5%
temperature calibration (molecule-level, frozen). During training, validation
NLL is evaluated every 10 epochs; the best-validation checkpoint is kept.
Temperature T is then fitted on the 5% temperature-calibration subset of that
same model, and the calibrated contrast is computed on the untouched 20% test
partition. Fixed-final-step models (b3_main_v2) serve as the baseline.

Outputs: b3_checkpoint_out/ckpt_di.parquet (p_stl_ckpt / p_mtl_ckpt per
molecule) + console comparison vs fixed-final results (3 instances x 3 seeds).
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *
from b3_main_experiment import MLP, fp_array, load_endpoints
from b3_main_v2 import global_split3, nll_loss, temp_scale

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)
OUT = os.path.join(os.path.dirname(__file__), 'b3_checkpoint_out')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(DATA_DIR)
VAL_EVERY = 10


def split_cal(cal_mols, seed):
    """Split the calibration partition into validation (50%) and temperature-cal (50%)."""
    rng = np.random.RandomState(seed + 999)
    arr = np.array(sorted(cal_mols))
    perm = rng.permutation(len(arr))
    cut = len(arr) // 2
    return set(arr[perm[:cut]].tolist()), set(arr[perm[cut:]].tolist())


def train_with_ckpt(X, y, val_tasks, seed, n_heads, head=0):
    """Train with validation-based checkpoint selection; return best model.
    val_tasks: list of (X_val_h, y_val_h) per head; the validation NLL is the
    mean over heads, each head evaluated on its own endpoint's validation
    partition (the same partition structure as training)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X.shape[1], n_heads=n_heads).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    val_data = [(torch.tensor(xv, dtype=torch.float32, device=DEVICE), yv)
                for xv, yv in val_tasks]
    n = len(Xt)
    best_nll, best_state = float('inf'), None
    model.train()
    for epoch in range(EPOCHS):
        for h in range(n_heads):
            for _ in range(8):
                bidx = torch.randint(0, n, (BATCH,), device=DEVICE)
                opt.zero_grad()
                loss = lossf(model(Xt[bidx], head=h), yt[bidx])
                loss.backward()
                opt.step()
        if (epoch + 1) % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vn = 0.0
                for h in range(n_heads):
                    Xvh, yvh = val_data[h % len(val_data)]
                    p = torch.sigmoid(model(Xvh, head=h)).cpu().numpy()
                    vn += nll_loss(yvh, p).mean()
                vn /= n_heads
            if vn < best_nll:
                best_nll, best_state = vn, {k: v.clone() for k, v in model.state_dict().items()}
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    return model, best_nll


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
        tr, ca, te = global_split3(all_mols, seed=SEED_BASE + inst)
        val_m, cal_m = split_cal(ca, SEED_BASE + inst)
        tr_idx, va_idx, ca_idx, te_idx = {}, {}, {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in tr])
            va_idx[ep] = np.array([i for i, c in enumerate(cl) if c in val_m])
            ca_idx[ep] = np.array([i for i, c in enumerate(cl) if c in cal_m])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in te])
            if len(tr_idx[ep]) > MAX_TRAIN_PER_ENDPOINT:
                rng = np.random.RandomState(SEED_BASE + inst)
                tr_idx[ep] = rng.choice(tr_idx[ep], MAX_TRAIN_PER_ENDPOINT, replace=False)
        for seed in range(3):
            seed_i = SEED_BASE + seed * 7 + inst
            tasks = [(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].values[tr_idx[ep]]) for ep in eps]
            # MTL: validation NLL averaged over heads, each head on its own endpoint's
            # validation partition (matches the training partition structure).
            mtl_val = [(fps[ep][va_idx[ep]], ep_data[ep]['Y'].values[va_idx[ep]]) for ep in eps]
            mtl, _ = train_with_ckpt(tasks[0][0], tasks[0][1], mtl_val, seed_i,
                                     n_heads=len(tasks), head=0)
            for h, ep in enumerate(eps):
                stl, _ = train_with_ckpt(tasks[h][0], tasks[h][1],
                                         [(fps[ep][va_idx[ep]], ep_data[ep]['Y'].values[va_idx[ep]])],
                                         seed_i, n_heads=1, head=0)
                X_te = torch.tensor(fps[ep][te_idx[ep]], dtype=torch.float32, device=DEVICE)
                y_te = ep_data[ep]['Y'].values[te_idx[ep]]
                X_ca = torch.tensor(fps[ep][ca_idx[ep]], dtype=torch.float32, device=DEVICE)
                y_ca = ep_data[ep]['Y'].values[ca_idx[ep]]
                with torch.no_grad():
                    p_stl = torch.sigmoid(stl(X_te, head=0)).cpu().numpy()
                    p_mtl = torch.sigmoid(mtl(X_te, head=h)).cpu().numpy()
                    p_stl_ca = torch.sigmoid(stl(X_ca, head=0)).cpu().numpy()
                    p_mtl_ca = torch.sigmoid(mtl(X_ca, head=h)).cpu().numpy()
                T_stl = minimize_scalar(lambda T: nll_loss(y_ca, temp_scale(p_stl_ca, T)).mean(),
                                        bounds=(0.2, 5.0), method='bounded').x
                T_mtl = minimize_scalar(lambda T: nll_loss(y_ca, temp_scale(p_mtl_ca, T)).mean(),
                                        bounds=(0.2, 5.0), method='bounded').x
                rows.append(pd.DataFrame({
                    'inst': inst, 'seed': seed, 'endpoint': ep,
                    'mol': np.array(canon_lists[ep])[te_idx[ep]], 'y': y_te,
                    'p_stl_ckpt': p_stl, 'p_mtl_ckpt': p_mtl,
                    'p_stl_cal': temp_scale(p_stl, T_stl),
                    'p_mtl_cal': temp_scale(p_mtl, T_mtl)}))
            print(f'[{time.time()-t0:.0f}s] inst={inst} seed={seed} done', flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(f'{OUT}/ckpt_di.parquet')
    # 汇总 vs fixed-final（v2）
    v2 = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'b3_v2_out/b3_v2_random_di.parquet'))
    v2 = v2[v2.inst.isin([0, 1, 2])]
    m = df.merge(v2[['inst', 'seed', 'endpoint', 'mol', 'y', 'd', 'd_cal']],
                 on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')
    d_ckpt = m.apply(lambda r: nll_loss(r['y'], r['p_stl_ckpt']).mean() -
                     nll_loss(r['y'], r['p_mtl_ckpt']).mean(), axis=1)
    d_ckpt_cal = m.apply(lambda r: nll_loss(r['y'], r['p_stl_cal']).mean() -
                         nll_loss(r['y'], r['p_mtl_cal']).mean(), axis=1)
    print(f'\n=== validation-selected checkpoints (3x3, random) ===')
    print(f'd_ckpt (raw)       = {d_ckpt.mean():+.4f}  (fixed-final: {m.d.mean():+.4f})')
    print(f'd_ckpt (calibrated)= {d_ckpt_cal.mean():+.4f}  (fixed-final: {m.d_cal.mean():+.4f})')
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
