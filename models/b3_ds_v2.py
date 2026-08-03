"""
B3 R2c P0-2: downsampling under the v2 confirmatory design (70/10/20 three-way
global allocation, 5 instances x 3 seeds, scaffold protocol).

The v1 downsampling numbers (CYP2C9 +0.30->+0.17 etc.) came from the old 80/20
two-way design and must not be mixed with v2 main numbers. This script re-runs
the intervention under the current design: the three CYP endpoints are
subsampled to 2,000 target-task training molecules (nested, class prevalence
preserved, calibration/test partitions unchanged, auxiliary tasks unchanged,
task-balanced schedule held constant), scaffold protocol only (the protocol
under which the intervention is reported).

Output: b3_v2_out/b3_v2_scaffold_ds_di.parquet (per-molecule d) + console summary.
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
sys.path.insert(0, os.path.dirname(__file__))
from b3_config import *
from b3_main_v2 import (DEVICE, OUT, load_endpoints, fp_array, global_split3,
                        MLP, train_model, predict_p, nll_loss, nearest_train_tanimoto,
                        T_BOUNDS, fit_T, temp_scale)

DS_ENDPOINTS = ['CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith']
DS_TRAIN_SIZE = 2000
DS_SEED = SEED_BASE + 77  # frozen downsampling sampler seed


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
    for inst in range(N_SPLIT_INSTANCES):
        train_set, cal_set, test_set = global_split3(
            all_mols, seed=SEED_BASE + inst, scaffold=True)
        tr_idx, ca_idx, te_idx = {}, {}, {}
        for ep in ep_data:
            cl = canon_lists[ep]
            tr_idx[ep] = np.array([i for i, c in enumerate(cl) if c in train_set])
            ca_idx[ep] = np.array([i for i, c in enumerate(cl) if c in cal_set])
            te_idx[ep] = np.array([i for i, c in enumerate(cl) if c in test_set])
        # downsampling: subsample target-task train to DS_TRAIN_SIZE (nested, fixed seed)
        for ep in DS_ENDPOINTS:
            rng = np.random.RandomState(DS_SEED + inst)
            if len(tr_idx[ep]) > DS_TRAIN_SIZE:
                tr_idx[ep] = rng.choice(tr_idx[ep], DS_TRAIN_SIZE, replace=False)
        print(f'[{time.time()-t0:.0f}s] ds inst={inst}: '
              f'CYP train sizes = {[len(tr_idx[e]) for e in DS_ENDPOINTS]}', flush=True)
        for seed in range(N_SEEDS):
            seed_i = SEED_BASE + seed * 7 + inst
            tasks = [(fps[ep][tr_idx[ep]], ep_data[ep]['Y'].values[tr_idx[ep]]) for ep in eps]
            mtl = train_model(tasks, seed_i)
            for h, ep in enumerate(eps):
                stl = train_model(tasks, seed_i, stl_ep=h)
                X_te = fps[ep][te_idx[ep]]
                y_te = ep_data[ep]['Y'].values[te_idx[ep]]
                X_ca = fps[ep][ca_idx[ep]]
                y_ca = ep_data[ep]['Y'].values[ca_idx[ep]]
                p_stl = predict_p(stl, X_te, head=0)
                p_mtl = predict_p(mtl, X_te, head=h)
                p_stl_ca = predict_p(stl, X_ca, head=0)
                p_mtl_ca = predict_p(mtl, X_ca, head=h)
                T_stl = fit_T(y_ca, p_stl_ca)
                T_mtl = fit_T(y_ca, p_mtl_ca)
                d = nll_loss(y_te, p_stl).mean(axis=0) - nll_loss(y_te, p_mtl).mean(axis=0)
                rows.append(pd.DataFrame({
                    'inst': inst, 'seed': seed, 'endpoint': ep,
                    'mol': np.array(canon_lists[ep])[te_idx[ep]], 'y': y_te,
                    'p_stl': p_stl, 'p_mtl': p_mtl,
                    'd': d, 'T_stl': T_stl, 'T_mtl': T_mtl}))
            print(f'[{time.time()-t0:.0f}s] ds inst={inst} seed={seed} done', flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(f'{OUT}/b3_v2_scaffold_ds_di.parquet')
    g = df.groupby('endpoint').d.mean()
    print(f'\n=== v2 scaffold downsampled: d_mean={df.d.mean():+.4f} ===')
    print(g.round(4).to_string())
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
