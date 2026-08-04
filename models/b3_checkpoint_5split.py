"""
B3 Sensitivity Addendum A (frozen 2026-08-04, protocol/sensitivity_addendum_v1.md):
5-split checkpoint-selection sensitivity.

Design: 5 split instances x 3 seeds, protocols grouped-random and
scaffold-grouped; the 10% calibration partition is split into 5% validation +
5% temperature-calibration (split_cal, seed + 999, unchanged). Every 10 epochs
the model state is saved and per-head validation NLL is recorded. Four rules
are evaluated on the saved checkpoints:

  1. fixed-final          : epoch 60 for all models
  2. global-vs-global     : MTL one common epoch (head-mean val NLL); the eight
                            STL select one common epoch (endpoint-mean val NLL)
  3. endpoint-vs-endpoint : each MTL head and its corresponding STL select the
                            epoch minimizing that endpoint's val NLL
  4. original hybrid      : MTL one common epoch; each STL its own endpoint
                            epoch (the previously reported post hoc rule)

MTL training is the true multi-task schedule: head h trains on endpoint h's
task (8 updates/head/epoch), matching b3_main_v2. Aggregation: seeds averaged
within split; the split instance is the statistical unit (5 units);
split-instance block bootstrap 95% CI. Calibration: strict per-model
temperatures fitted on the 5% temperature-calibration subset (bounds
T_BOUNDS), applied to the untouched test partition.

Outputs (per addendum):
  results/checkpoint_5split_summary.csv
  results/checkpoint_selected_epochs.csv
  results/checkpoint_val_curves.csv
  figures/fig_checkpoint_learning_curves.pdf
  figures/fig_checkpoint_rule_comparison.pdf
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 9.5, 'axes.labelsize': 9.5, 'legend.fontsize': 8.5})

sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *
from b3_main_experiment import MLP, fp_array, load_endpoints
from b3_main_v2 import global_split3, nll_loss, temp_scale, fit_T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'b3_ckpt5_out')
FIG = '/home/as/vllm/cell/paper/figures'
os.makedirs(OUT, exist_ok=True)

EPOCHS = 60
VAL_EVERY = 10
RULES = ['fixed-final', 'global-vs-global', 'endpoint-vs-endpoint', 'hybrid']


def split_cal(cal_mols, seed):
    """Split the calibration partition into validation (50%) and temperature-cal (50%)."""
    rng = np.random.RandomState(seed + 999)
    arr = np.array(sorted(cal_mols))
    perm = rng.permutation(len(arr))
    cut = len(arr) // 2
    return set(arr[perm[:cut]].tolist()), set(arr[perm[cut:]].tolist())


def train_mtl_save_ckpts(tasks, val_tasks, seed):
    """True MTL: head h trained on tasks[h]; returns (ckpts, val_nll_heads)."""
    n_heads = len(tasks)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(tasks[0][0].shape[1], n_heads=n_heads).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    data = [(torch.tensor(X, dtype=torch.float32, device=DEVICE),
             torch.tensor(y, dtype=torch.float32, device=DEVICE)) for X, y in tasks]
    val_data = [(torch.tensor(xv, dtype=torch.float32, device=DEVICE), yv) for xv, yv in val_tasks]
    ckpts, vn = {}, {}
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
        if (epoch + 1) % VAL_EVERY == 0:
            model.eval()
            vh = np.zeros(n_heads)
            with torch.no_grad():
                for h in range(n_heads):
                    Xvh, yvh = val_data[h]
                    p = torch.sigmoid(model(Xvh, head=h)).cpu().numpy()
                    vh[h] = nll_loss(yvh, p).mean()
            vn[epoch + 1] = vh
            ckpts[epoch + 1] = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()
    return ckpts, vn


def train_stl_save_ckpts(X, y, val_task, seed):
    """Single-head STL; returns (ckpts, val_nll {epoch: float})."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X.shape[1], n_heads=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(val_task[0], dtype=torch.float32, device=DEVICE)
    yv = val_task[1]
    n = len(Xt)
    ckpts, vn = {}, {}
    model.train()
    for epoch in range(EPOCHS):
        for _ in range(8):
            bidx = torch.randint(0, n, (BATCH,), device=DEVICE)
            opt.zero_grad()
            loss = lossf(model(Xt[bidx], head=0), yt[bidx])
            loss.backward()
            opt.step()
        if (epoch + 1) % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xv, head=0)).cpu().numpy()
            vn[epoch + 1] = nll_loss(yv, p).mean()
            ckpts[epoch + 1] = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()
    return ckpts, vn


def select_epochs(vn_mtl, vn_stl, rule):
    """vn_mtl: {epoch: np.array(8)}; vn_stl: {ep: {epoch: float}}.
    Returns (mtl_epochs, stl_epochs) as np.array(8) each (per-head epochs)."""
    evs = sorted(vn_mtl.keys())
    if rule == 'fixed-final':
        return np.full(8, evs[-1]), np.full(8, evs[-1])
    mtl_macro = {e: vn_mtl[e].mean() for e in evs}
    stl_macro = {e: np.mean([vn_stl[ep][e] for ep in range(8)]) for e in evs}
    mtl_common = min(mtl_macro, key=mtl_macro.get)
    stl_common = min(stl_macro, key=stl_macro.get)
    stl_own = np.array([min(vn_stl[ep], key=vn_stl[ep].get) for ep in range(8)])
    if rule == 'global-vs-global':
        return np.full(8, mtl_common), np.full(8, stl_common)
    if rule == 'hybrid':
        return np.full(8, mtl_common), stl_own
    if rule == 'endpoint-vs-endpoint':
        mtl_own = np.array([min(evs, key=lambda e: vn_mtl[e][ep]) for ep in range(8)])
        return mtl_own, stl_own
    raise ValueError(rule)


def predict_endpoint(model_ckpts, epoch, X_te, head, n_heads):
    model = MLP(X_te.shape[1], n_heads=n_heads).to(DEVICE)
    model.load_state_dict(model_ckpts[epoch])
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(Xt, head=head)).cpu().numpy()


def main():
    t0 = time.time()
    ep_data = load_endpoints()
    all_mols = set()
    for ep, df in ep_data.items():
        all_mols |= set(df['canon'])
    fps = {ep: fp_array(df['canon'].tolist()) for ep, df in ep_data.items()}
    canon_lists = {ep: df['canon'].tolist() for ep, df in ep_data.items()}
    eps = list(ep_data.keys())
    n_ep = len(eps)

    delta_rows = []   # protocol, rule, inst, seed, endpoint, delta, delta_cal
    epoch_rows = []   # protocol, inst, seed, rule, model, endpoint, epoch
    val_rows = []     # protocol, inst, seed, epoch, mtl_macro, stl_macro

    for protocol in ['random', 'scaffold']:
        for inst in range(5):
            train_set, cal_set, test_set = global_split3(
                all_mols, seed=SEED_BASE + inst, scaffold=(protocol == 'scaffold'))
            val_set, tc_set = split_cal(cal_set, SEED_BASE + inst)
            tr_idx, va_idx, tc_idx, te_idx = {}, {}, {}, {}
            for ep in eps:
                cl = canon_lists[ep]
                tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in train_set])
                va_idx[ep] = np.array([i for i, c in enumerate(cl) if c in val_set])
                tc_idx[ep] = np.array([i for i, c in enumerate(cl) if c in tc_set])
                te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in test_set])
                if len(tr_idx[ep]) > MAX_TRAIN_PER_ENDPOINT:
                    rng = np.random.RandomState(SEED_BASE + inst)
                    tr_idx[ep] = rng.choice(tr_idx[ep], MAX_TRAIN_PER_ENDPOINT, replace=False)
            for seed in range(3):
                seed_i = SEED_BASE + seed * 7 + inst
                tasks = [(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].values[tr_idx[ep]]) for ep in eps]
                val_tasks = [(fps[ep][va_idx[ep]], ep_data[ep]['Y'].values[va_idx[ep]]) for ep in eps]
                tc_tasks = [(fps[ep][tc_idx[ep]], ep_data[ep]['Y'].values[tc_idx[ep]]) for ep in eps]
                te_tasks = [(fps[ep][te_idx[ep]], ep_data[ep]['Y'].values[te_idx[ep]]) for ep in eps]

                ck_mtl, vn_mtl = train_mtl_save_ckpts(tasks, val_tasks, seed_i)
                ck_stl, vn_stl = {}, {}
                for h, ep in enumerate(eps):
                    ck_stl[ep], vn_stl[h] = train_stl_save_ckpts(
                        tasks[h][0], tasks[h][1], val_tasks[h], seed_i)

                for e in sorted(vn_mtl.keys()):
                    val_rows.append({'protocol': protocol, 'inst': inst, 'seed': seed,
                                     'epoch': e,
                                     'mtl_macro': vn_mtl[e].mean(),
                                     'stl_macro': np.mean([vn_stl[h][e] for h in range(n_ep)])})

                for rule in RULES:
                    mtl_eps, stl_eps = select_epochs(vn_mtl, vn_stl, rule)
                    for h, ep in enumerate(eps):
                        y_te = te_tasks[h][1]
                        p_stl = predict_endpoint(ck_stl[ep], stl_eps[h], te_tasks[h][0], 0, 1)
                        p_mtl = predict_endpoint(ck_mtl, mtl_eps[h], te_tasks[h][0], h, n_ep)
                        # temperature calibration on the temp-cal subset
                        y_tc = tc_tasks[h][1]
                        p_stl_tc = predict_endpoint(ck_stl[ep], stl_eps[h], tc_tasks[h][0], 0, 1)
                        p_mtl_tc = predict_endpoint(ck_mtl, mtl_eps[h], tc_tasks[h][0], h, n_ep)
                        T_stl = fit_T(y_tc, p_stl_tc)
                        T_mtl = fit_T(y_tc, p_mtl_tc)
                        d = nll_loss(y_te, p_stl).mean() - nll_loss(y_te, p_mtl).mean()
                        d_cal = (nll_loss(y_te, temp_scale(p_stl, T_stl)).mean()
                                 - nll_loss(y_te, temp_scale(p_mtl, T_mtl)).mean())
                        delta_rows.append({'protocol': protocol, 'rule': rule,
                                           'inst': inst, 'seed': seed, 'endpoint': ep,
                                           'delta': d, 'delta_cal': d_cal})
                        epoch_rows.append({'protocol': protocol, 'inst': inst, 'seed': seed,
                                           'rule': rule, 'model': 'MTL', 'endpoint': ep,
                                           'epoch': int(mtl_eps[h])})
                        epoch_rows.append({'protocol': protocol, 'inst': inst, 'seed': seed,
                                           'rule': rule, 'model': 'STL', 'endpoint': ep,
                                           'epoch': int(stl_eps[h])})
                print(f'[{time.time()-t0:.0f}s] {protocol} inst={inst} seed={seed} done', flush=True)

    df_d = pd.DataFrame(delta_rows)
    df_e = pd.DataFrame(epoch_rows)
    df_v = pd.DataFrame(val_rows)
    df_d.to_csv(os.path.join(OUT, 'checkpoint_5split_deltas.csv'), index=False)
    df_e.to_csv(os.path.join(OUT, 'checkpoint_selected_epochs.csv'), index=False)
    df_v.to_csv(os.path.join(OUT, 'checkpoint_val_curves.csv'), index=False)

    # ---- aggregate: split-level, seeds averaged within split ----
    per = []
    for protocol in ['random', 'scaffold']:
        for rule in RULES:
            sub = df_d[(df_d.protocol == protocol) & (df_d.rule == rule)]
            em = sub.groupby(['inst', 'seed'])[['delta', 'delta_cal']].mean()
            sl = em.groupby('inst').mean()
            for inst_ in sl.index:
                per.append({'rule': rule, 'inst': int(inst_), 'protocol': protocol,
                            'delta': sl.loc[inst_, 'delta'],
                            'delta_cal': sl.loc[inst_, 'delta_cal']})
    per_df = pd.DataFrame(per)
    wide = per_df.pivot_table(index=['rule', 'inst'], columns='protocol',
                              values=['delta', 'delta_cal']).reset_index()
    out = []
    rng = np.random.RandomState(7)
    for rule in RULES:
        g = wide[wide.rule == rule].sort_values('inst')
        d_r = g[('delta', 'random')].values
        d_s = g[('delta', 'scaffold')].values
        dr_c = g[('delta_cal', 'random')].values
        ds_c = g[('delta_cal', 'scaffold')].values
        gk = d_s - d_r
        gk_cal = ds_c - dr_c
        boots = [gk[rng.choice(5, 5, replace=True)].mean() for _ in range(3000)]
        ci = np.percentile(boots, [2.5, 97.5])
        out.append({'rule': rule,
                    'delta_random': d_r.mean(), 'delta_scaffold': d_s.mean(),
                    'gamma': gk.mean(),
                    'delta_cal_random': dr_c.mean(), 'delta_cal_scaffold': ds_c.mean(),
                    'gamma_cal': gk_cal.mean(),
                    'positive_splits_raw': int((gk > 0).sum()),
                    'positive_splits_cal': int((gk_cal > 0).sum()),
                    'gamma_ci_lo': ci[0], 'gamma_ci_hi': ci[1],
                    'split_gammas': ';'.join(f'{v:+.3f}' for v in gk),
                    'split_gammas_cal': ';'.join(f'{v:+.3f}' for v in gk_cal)})
    summary_df = pd.DataFrame(out)
    summary_df.to_csv(os.path.join(OUT, 'checkpoint_5split_summary.csv'), index=False)
    print('\n=== checkpoint_5split_summary ===')
    print(summary_df.to_string(index=False))

    # ---- figures ----
    # learning curves (validation NLL per epoch, mean over 15 runs)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, proto in zip(axes, ['random', 'scaffold']):
        v = df_v[df_v.protocol == proto]
        grp = v.groupby('epoch')[['mtl_macro', 'stl_macro']].agg(['mean', 'std'])
        e = grp.index.values
        for col, c in [('mtl_macro', '#4C72B0'), ('stl_macro', '#DD8452')]:
            ax.errorbar(e, grp[(col, 'mean')], yerr=1.96 * grp[(col, 'std')] / np.sqrt(15),
                        fmt='o-', ms=4, capsize=2, color=c, label=col.split('_')[0].upper())
        ax.set_xlabel('epoch')
        ax.set_ylabel('validation NLL')
        ax.set_title(proto, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_checkpoint_learning_curves.pdf')
    plt.close(fig)

    # rule comparison: 5 split-level Gamma per rule (raw and calibrated)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for i, rule in enumerate(RULES):
        g = wide[wide.rule == rule].sort_values('inst')
        gk = g[('delta', 'scaffold')].values - g[('delta', 'random')].values
        jit = rng.uniform(-0.10, 0.10, 5)
        ax.scatter(np.full(5, i) + jit, gk, s=30, color='#4C72B0', alpha=0.85,
                   edgecolors='white', linewidths=0.5)
        ax.plot(i, gk.mean(), marker='D', ms=11, color='#C44E52',
                markeredgecolor='black', markeredgewidth=0.7)
        ax.annotate(f'{gk.mean():+.3f}', (i, gk.mean()), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=8.5, color='#C44E52')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['fixed-final', 'global-vs-\nglobal', 'endpoint-vs-\nendpoint', 'hybrid'], fontsize=8)
    ax.set_ylabel(r'$\Gamma$ (scaffold $-$ random)')
    ax.set_title('Split-level protocol contrast by checkpoint rule (5 splits)', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_checkpoint_rule_comparison.pdf')
    plt.close(fig)
    print('figures saved')
    print(f'\nTOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
