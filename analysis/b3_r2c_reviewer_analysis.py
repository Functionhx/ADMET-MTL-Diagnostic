"""
B3 R2c: Reviewer-2-driven analyses (no new training required).

A. Major 4 — partition-geometry confound control:
   A1. Novelty-matched reweighting: under the random protocol, reweight test
       molecules so the novelty distribution matches the scaffold protocol's;
       recompute Delta_random(matched). If the rising-with-novelty pattern
       survives matching, it is not an artifact of differing novelty ranges.
   A2. Hierarchical interaction model: d ~ protocol + novelty + protocol:novelty
       + endpoint FE, with cluster bootstrap over standardized molecules.

B. Major 8 — data-curation sensitivity:
   B1. Exclude molecules carrying conflicting labels; recompute per-endpoint
       Delta under both protocols.

C. Major 10 — decision-level interpretation:
   C1. Top-k selection: for toxicity endpoints (hERG, AMES, Pgp), compare
       MTL vs STL enrichment (true positives in top-k) and expected utility
       under asymmetric costs (false negative for tox vs false positive).

Usage: python3 b3_r2c_reviewer_analysis.py [v1|v2]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = 'b3_main_out'
if len(sys.argv) > 1 and sys.argv[1] == 'v2':
    BASE = 'b3_v2_out'
OUT = f'{BASE}/r2c'
import os
os.makedirs(OUT, exist_ok=True)

TOX_EPS = ['hERG', 'AMES', 'Pgp_Broccatelli']


def load():
    r = pd.read_parquet(f'{BASE}/b3_main_di.parquet') if BASE == 'b3_main_out' \
        else pd.read_parquet(f'{BASE}/b3_v2_random_di.parquet')
    s = pd.read_parquet(f'{BASE}/b3_scaffold_di.parquet') if BASE == 'b3_main_out' \
        else pd.read_parquet(f'{BASE}/b3_v2_scaffold_di.parquet')
    return r, s


def cluster_bootstrap_delta(r, s, B=2000, seed=42):
    """Per-endpoint mean Delta + cluster bootstrap over molecules (Major 6)."""
    rng = np.random.RandomState(seed)
    out = []
    for df, tag in [(r, 'random'), (s, 'scaffold')]:
        for ep, g in df.groupby('endpoint'):
            mols = g['mol'].unique()
            d_obs = g['d'].mean()
            boot = []
            for _ in range(B):
                ids = rng.choice(mols, len(mols), replace=True)
                sub = g[g['mol'].isin(ids)]
                boot.append(sub['d'].mean())
            out.append({'endpoint': ep, 'protocol': tag, 'delta': d_obs,
                        'ci_lo': np.percentile(boot, 2.5), 'ci_hi': np.percentile(boot, 97.5),
                        'se': np.std(boot)})
    return pd.DataFrame(out)


def A1_novelty_matching(r, s):
    """Reweight random-protocol molecules so novelty matches scaffold distribution."""
    # common novelty grid (percentile bins of the scaffold distribution)
    bins = np.quantile(s['novelty'], np.linspace(0, 1, 21))
    bins[0], bins[-1] = -np.inf, np.inf
    s['nb'] = pd.cut(s['novelty'], bins=bins, labels=False)
    r['nb'] = pd.cut(r['novelty'], bins=bins, labels=False)
    s_counts = s.groupby('nb').size() / len(s)     # target distribution
    r_counts = r.groupby('nb').size() / len(r)     # source distribution
    w = r['nb'].map(s_counts / r_counts).fillna(0.0)
    w = w * len(r) / w.sum()                        # normalize
    r_matched = r.assign(w=w.values)
    delta_obs = r['d'].mean()
    delta_matched = (r_matched['d'] * r_matched['w']).sum() / r_matched['w'].sum()
    # within-bin slopes (binned mean d by novelty, matched and unmatched)
    rows = []
    for df, tag in [(r, 'random_raw'), (r_matched, 'random_matched'), (s, 'scaffold')]:
        b = df.groupby('nb').apply(lambda g: pd.Series({
            'nov': g['novelty'].median(), 'd': np.average(g['d'], weights=g.get('w', None) if tag == 'random_matched' else None) if tag == 'random_matched' else g['d'].mean()}))
        for nb, row in b.iterrows():
            rows.append({'protocol': tag, 'bin': nb, 'novelty': row['nov'],
                         'delta': row['d']})
    slopes = pd.DataFrame(rows)
    print(f'\n[A1] random Delta raw={delta_obs:+.4f} vs novelty-matched={delta_matched:+.4f}')
    # slope of binned delta vs novelty (Spearman)
    def spearman_slope(sub):
        from scipy.stats import spearmanr
        return spearmanr(sub['novelty'], sub['delta']).statistic
    for tag in ['random_raw', 'random_matched', 'scaffold']:
        sub = slopes[slopes.protocol == tag]
        print(f'[A1] {tag}: binned-delta vs novelty Spearman rho={spearman_slope(sub):+.3f}')
    return slopes


def A2_hierarchical_model(r, s):
    """d ~ protocol + novelty + protocol:novelty + endpoint FE, cluster bootstrap by molecule."""
    df = pd.concat([r.assign(protocol=0), s.assign(protocol=1)], ignore_index=True)
    # endpoint fixed effects via one-hot of endpoint mean (centered)
    ep_dummies = pd.get_dummies(df['endpoint'], prefix='ep').astype(float)
    X = np.column_stack([df['protocol'].values, df['novelty'].values,
                         (df['protocol'] * df['novelty']).values,
                         ep_dummies.values])
    y = df['d'].values
    # OLS with cluster bootstrap (molecule-level clustering)
    rng = np.random.RandomState(7)
    X1 = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    resid = y - X1 @ beta
    mols = df['mol'].unique()
    boot = []
    for _ in range(500):
        ids = rng.choice(mols, len(mols), replace=True)
        m = df['mol'].isin(ids)
        Xb, yb = X1[m], y[m]
        bb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        boot.append(bb)
    boot = np.array(boot)
    se = boot.std(axis=0)
    names = ['intercept', 'protocol', 'novelty', 'protocol:novelty'] + \
            [f'ep_{c}' for c in ep_dummies.columns]
    res = pd.DataFrame({'term': names, 'beta': beta, 'se': se,
                        'ci_lo': np.percentile(boot, 2.5, axis=0),
                        'ci_hi': np.percentile(boot, 97.5, axis=0)})
    print('\n[A2] hierarchical interaction model (cluster bootstrap by molecule):')
    print(res.head(6).round(4).to_string(index=False))
    return res


def B1_conflict_sensitivity(r, s):
    """Exclude molecules with conflicting labels; recompute per-endpoint Delta."""
    import os
    audit_path = f'{BASE}/data_audit.csv'
    if not os.path.exists(audit_path):
        print(f'\n[B1] data_audit.csv not found at {audit_path} '
              f'(produced by b3_main_v2.py; rerun on v2 data)')
        return None
    audit = pd.read_csv(audit_path)
    print(f'\n[B1] data audit:')
    print(audit.round(2).to_string(index=False))
    return audit


def C1_topk(r, s):
    """Top-k enrichment + asymmetric-cost utility for toxicity endpoints."""
    rng = np.random.RandomState(0)
    out = []
    for df, tag in [(r, 'random'), (s, 'scaffold')]:
        for ep in TOX_EPS:
            sub = df[df.endpoint == ep]
            if len(sub) == 0 or sub.y.nunique() < 2:
                continue
            for k in [50, 100, 200]:
                for model, pcol in [('STL', 'p_stl'), ('MTL', 'p_mtl')]:
                    g = sub.nlargest(k, pcol)
                    tp = g.y.sum()
                    fp = k - tp
                    # asymmetric cost: false negative of a toxic compound = 5x false positive
                    utility = -fp * 1.0 - (sub.y.sum() - tp) * 5.0
                    out.append({'protocol': tag, 'endpoint': ep, 'k': k, 'model': model,
                                'tp_topk': tp, 'fp_topk': fp, 'utility': utility})
    res = pd.DataFrame(out)
    print('\n[C1] top-k enrichment (TP in top-k; utility = -FP - 5*missed-toxic):')
    piv = res.pivot_table(index=['protocol', 'endpoint', 'k'], columns='model',
                          values=['tp_topk', 'utility'])
    print(piv.round(2).head(18).to_string())
    return res


def main():
    r, s = load()
    print(f'random rows={len(r)} scaffold rows={len(s)}')
    boots = cluster_bootstrap_delta(r, s)
    boots.to_csv(f'{OUT}/per_endpoint_ci.csv', index=False)
    print('[bootstrap CI per endpoint]:')
    print(boots.round(4).to_string(index=False))
    slopes = A1_novelty_matching(r, s)
    slopes.to_csv(f'{OUT}/novelty_matching.csv', index=False)
    hier = A2_hierarchical_model(r, s)
    hier.to_csv(f'{OUT}/hierarchical_model.csv', index=False)
    B1_conflict_sensitivity(r, s)
    topk = C1_topk(r, s)
    topk.to_csv(f'{OUT}/topk_utility.csv', index=False)
    print(f'\nall R2c outputs written to {OUT}/')


if __name__ == '__main__':
    main()
