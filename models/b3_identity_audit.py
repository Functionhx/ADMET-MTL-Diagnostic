"""
B3 Sensitivity Addendum B (frozen 2026-08-04, protocol/sensitivity_addendum_v1.md):
chemical-entity identity collision audit.

Identity levels:
  L0: exact RDKit canonical isomeric SMILES (primary key, as used)
  L1: standardized parent key (Cleanup -> FragmentParent -> Uncharge ->
      canonical tautomer -> canonical isomeric SMILES)
  L2: InChIKey connectivity block (first block) - upper-bound related-entity
      audit only, never used as the identity definition (may merge stereoisomers)

Audit per (protocol, instance) over the global unique-molecule pool: keys,
collision groups spanning train/cal/test, test observations whose key also
appears in train or calibration (counts, fractions, per-endpoint breakdown).

No-retrain filtering sensitivity: from the existing paired prediction tables
(b3_v2_out), drop test observations whose L1 key appears in the train or
calibration partition of the same protocol instance, recompute raw and
calibrated Delta/Gamma at the split level with block bootstrap CI.

Outputs:
  results/chemical_identity_collision_audit.csv
  results/chemical_identity_collision_by_endpoint.csv
  results/chemical_identity_filtered_contrasts.csv
"""
import os, sys, time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize as rms

sys.path.insert(0, os.path.dirname(__file__))
from b3_config import SEED_BASE
from b3_main_experiment import load_endpoints
from b3_main_v2 import global_split3, nll_loss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'b3_idaudit_out')
os.makedirs(OUT, exist_ok=True)

_cleanup = rms.Cleanup
_uncharger = rms.Uncharger()
_te = rms.TautomerEnumerator()


def std_parent_key(smi):
    """L1: Cleanup -> FragmentParent -> Uncharge -> canonical tautomer ->
    canonical isomeric SMILES. Returns '' for unparseable input."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ''
        mol = _cleanup(mol)
        mol = rms.FragmentParent(mol)
        mol = _uncharger.uncharge(mol)
        mol = _te.Canonicalize(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ''


def inchi_block(smi):
    """L2: first block of the standardized parent's InChIKey."""
    k = std_parent_key(smi)
    if not k:
        return ''
    try:
        ik = inchi.InchiToInchiKey(Chem.MolToInchi(Chem.MolFromSmiles(k)))
        return ik.split('-')[0]
    except Exception:
        return ''


def main():
    t0 = time.time()
    ep_data = load_endpoints()
    all_mols = set()
    for ep, df in ep_data.items():
        all_mols |= set(df['canon'])
    eps = list(ep_data.keys())
    print(f'unique molecules: {len(all_mols)}', flush=True)

    # global key maps
    l1 = {m: std_parent_key(m) for m in all_mols}
    l2 = {m: inchi_block(m) for m in all_mols}
    l1_fail = sum(1 for m in all_mols if not l1[m])
    print(f'L1 standardization failures (unparseable): {l1_fail}', flush=True)

    audit_rows = []
    by_ep_rows = []
    for protocol in ['random', 'scaffold']:
        for inst in range(5):
            train_set, cal_set, test_set = global_split3(
                all_mols, seed=SEED_BASE + inst, scaffold=(protocol == 'scaffold'))
            for level, keymap in [('L1', l1), ('L2', l2)]:
                tr_k = set(keymap[m] for m in train_set if keymap[m])
                ca_k = set(keymap[m] for m in cal_set if keymap[m])
                te_k = set(keymap[m] for m in test_set if keymap[m])
                # collision groups spanning partitions
                groups = {}
                for m in all_mols:
                    k = keymap[m]
                    if not k:
                        continue
                    groups.setdefault(k, {'tr': 0, 'ca': 0, 'te': 0})
                    if m in train_set:
                        groups[k]['tr'] += 1
                    if m in cal_set:
                        groups[k]['ca'] += 1
                    if m in test_set:
                        groups[k]['te'] += 1
                n_col = sum(1 for k in groups
                            if (groups[k]['tr'] > 0) + (groups[k]['ca'] > 0) + (groups[k]['te'] > 0) > 1)
                # test observations whose key is in train / cal
                te_in_tr = 0
                te_in_ca = 0
                for m in test_set:
                    k = keymap[m]
                    if k in tr_k:
                        te_in_tr += 1
                    if k in ca_k:
                        te_in_ca += 1
                audit_rows.append({'protocol': protocol, 'instance': inst, 'level': level,
                                   'n_distinct_keys': len(groups),
                                   'n_collision_groups': n_col,
                                   'test_molecules': len(test_set),
                                   'test_in_train': te_in_tr,
                                   'test_in_train_frac': te_in_tr / len(test_set),
                                   'test_in_cal': te_in_ca,
                                   'test_in_cal_frac': te_in_ca / len(test_set)})
                # per-endpoint breakdown (test molecules only)
                for ep in eps:
                    te_mols = set(ep_data[ep]['canon']) & test_set
                    n_tr = sum(1 for m in te_mols if keymap[m] and keymap[m] in tr_k)
                    n_ca = sum(1 for m in te_mols if keymap[m] and keymap[m] in ca_k)
                    by_ep_rows.append({'protocol': protocol, 'instance': inst, 'level': level,
                                       'endpoint': ep, 'test_mols': len(te_mols),
                                       'test_in_train': n_tr, 'test_in_cal': n_ca})
        print(f'[{time.time()-t0:.0f}s] {protocol} audit done', flush=True)

    audit = pd.DataFrame(audit_rows)
    by_ep = pd.DataFrame(by_ep_rows)
    audit.to_csv(os.path.join(OUT, 'chemical_identity_collision_audit.csv'), index=False)
    by_ep.to_csv(os.path.join(OUT, 'chemical_identity_collision_by_endpoint.csv'), index=False)
    print('\n=== collision audit (L1) ===')
    print(audit[audit.level == 'L1'].to_string(index=False))

    # ---- no-retrain filtering sensitivity (Level 1) ----
    filt_rows = []
    for protocol in ['random', 'scaffold']:
        df = pd.read_parquet(os.path.join(HERE, f'b3_v2_out/b3_v2_{protocol}_di.parquet'))
        df = df[['inst', 'seed', 'endpoint', 'mol', 'y', 'd', 'd_cal']]
        df['l1'] = df.mol.map(l1)
        for inst in range(5):
            train_set, cal_set, test_set = global_split3(
                all_mols, seed=SEED_BASE + inst, scaffold=(protocol == 'scaffold'))
            tr_k = set(l1[m] for m in train_set if l1[m])
            ca_k = set(l1[m] for m in cal_set if l1[m])
            sub = df[df.inst == inst]
            mask = sub.l1.isin(tr_k | ca_k)
            n_drop = int(mask.sum())
            keep = sub[~mask]
            # per-endpoint means -> endpoint-macro
            def emacro(g):
                return g.groupby('endpoint')[['d', 'd_cal']].mean().mean()
            d_all = emacro(sub)
            d_filt = emacro(keep)
            filt_rows.append({'protocol': protocol, 'instance': inst,
                              'test_obs': len(sub), 'dropped_obs': n_drop,
                              'dropped_frac': n_drop / len(sub),
                              'delta_all': d_all.d, 'delta_cal_all': d_all.d_cal,
                              'delta_filtered': d_filt.d, 'delta_cal_filtered': d_filt.d_cal})
    filt = pd.DataFrame(filt_rows)
    filt.to_csv(os.path.join(OUT, 'chemical_identity_filtered_contrasts.csv'), index=False)
    # Gamma before/after per protocol
    print('\n=== filtered contrasts ===')
    for protocol in ['random', 'scaffold']:
        g = filt[filt.protocol == protocol]
        print(f'[{protocol}] Delta all={g.delta_all.mean():+.4f} filtered={g.delta_filtered.mean():+.4f} | '
              f'Delta_cal all={g.delta_cal_all.mean():+.4f} filtered={g.delta_cal_filtered.mean():+.4f}')
    # Gamma (scaffold - random) at split level
    wr = filt[filt.protocol == 'random'].set_index('instance')
    ws = filt[filt.protocol == 'scaffold'].set_index('instance')
    idx = sorted(set(wr.index) & set(ws.index))
    gk = (ws.loc[idx, 'delta_filtered'].values - wr.loc[idx, 'delta_filtered'].values)
    gk_all = (ws.loc[idx, 'delta_all'].values - wr.loc[idx, 'delta_all'].values)
    rng = np.random.RandomState(7)
    def boot(v):
        b = [v[rng.choice(len(v), len(v), replace=True)].mean() for _ in range(3000)]
        return np.percentile(b, [2.5, 97.5])
    print(f'Gamma all       = {gk_all.mean():+.4f} CI{boot(gk_all).round(3)}  pos splits {(gk_all > 0).sum()}/5')
    print(f'Gamma filtered  = {gk.mean():+.4f} CI{boot(gk).round(3)}  pos splits {(gk > 0).sum()}/5')
    print(f'[identity audit done in {time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
