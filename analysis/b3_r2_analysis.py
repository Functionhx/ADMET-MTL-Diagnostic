"""
B3 R2: Transfer-window analysis on confirmatory data (R1 output).

Reads b3_main_di.parquet (molecule-level d_i, novelty, endpoint, inst, seed).
Fits spline of d on novelty (per protocol-equivalent aggregation), checks:
- per-endpoint aggregate deltas
- d vs novelty bins (continuous axis)
- window criterion (interior max, derivative sign change)
- ranking stability across instances (Kendall tau)
Also: log-loss decomposition — is MTL worse in log-loss but better in AUC?
"""
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

OUT = '/home/as/vllm/cell/idea-stage/b3_main_out'


def main():
    df = pd.read_parquet(f'{OUT}/b3_main_di.parquet')
    print(f'molecules: {len(df)}, endpoints: {df.endpoint.nunique()}, '
          f'instances: {df.inst.nunique()}, seeds: {df.seed.nunique()}')
    print(f'overall mean d = {df.d.mean():+.4f} (log-loss advantage MTL-STL; '
          f'positive = MTL better)')

    # Per-endpoint aggregates
    print('\n=== per-endpoint mean d (across inst x seed) ===')
    agg = df.groupby('endpoint').d.agg(['mean', 'std', 'count'])
    print(agg.round(4).sort_values('mean'))

    # d vs novelty bins (continuous axis, primary)
    print('\n=== d by novelty bin (nearest-train Tanimoto) ===')
    df['nov_bin'] = pd.qcut(df['novelty'], 8, labels=False, duplicates='drop')
    nb = df.groupby('nov_bin').agg(mean_d=('d', 'mean'), n=('d', 'size'),
                                   nov_med=('novelty', 'median'))
    print(nb.round(4))

    # Spline fit: d ~ novelty (pooled, all instances/seeds)
    print('\n=== spline: d ~ novelty (pooled) ===')
    x = df['novelty'].values
    y = df['d'].values
    order = np.argsort(x)
    for s in [1e3, 1e4, 1e5, 1e6]:
        spl = UnivariateSpline(x[order], y[order], k=3, s=s)
        grid = np.linspace(x.min(), x.max(), 100)
        vals = spl(grid)
        imax = grid[np.argmax(vals)]
        print(f'  s={s:.0e}: knots={len(spl.get_knots())}, '
              f'val range=[{vals.min():+.4f},{vals.max():+.4f}], argmax={imax:.3f}')

    # Ranking stability of endpoint transfer gains across instances
    print('\n=== endpoint transfer-gain ranking stability across instances ===')
    piv = df.groupby(['inst', 'endpoint']).d.mean().unstack()
    taus = []
    for a in range(piv.shape[0]):
        for b in range(a + 1, piv.shape[0]):
            tau, p = kendalltau(piv.iloc[a].rank(), piv.iloc[b].rank())
            taus.append(tau)
    print(f'pairwise Kendall tau (instances): mean={np.mean(taus):.3f}, '
          f'min={np.min(taus):.3f}, max={np.max(taus):.3f}')

    # Per-instance mean AUC of MTL vs STL? not available in parquet (log-loss only)
    # Check class-conditional log-loss
    print('\n=== class-conditional log-loss (per molecule, pooled) ===')
    for cls, name in [(1, 'positive'), (0, 'negative')]:
        sub = df[df['y'] == cls]
        print(f'  {name}: n={len(sub)}, mean d={sub.d.mean():+.4f}')

    # Corroboration: log-loss vs prevalence
    print('\n=== d vs endpoint size (sparsity conditioning) ===')
    sz = df.groupby('endpoint').size()
    md = df.groupby('endpoint').d.mean()
    print(pd.concat([sz.rename('n_mol'), md.rename('mean_d')], axis=1).round(4))


if __name__ == '__main__':
    main()
