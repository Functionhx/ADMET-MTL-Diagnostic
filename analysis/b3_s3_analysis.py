"""
B3 S3: Within-endpoint continuous novelty analysis (outline review fix C2).
- Within-endpoint d ~ novelty regression (endpoint fixed effects)
- Endpoint size x novelty interaction
- Unique molecule counts (hygiene)
- Protocol interaction check C0: random vs scaffold protocols
Reads b3_main_di.parquet (random, R1b) + b3_scaffold_di.parquet (scaffold, supplement).
"""
import numpy as np
import pandas as pd
from scipy import stats

OUT = '/home/as/vllm/cell/idea-stage/b3_main_out'


def within_endpoint_analysis(df, tag):
    print(f'\n=== {tag}: within-endpoint continuous novelty ===')
    # endpoint-level fixed effects + novelty slope (molecule-level OLS with endpoint dummies)
    eps = df['endpoint'].astype('category')
    X = pd.get_dummies(eps, prefix='ep', drop_first=True).astype(float)
    X['novelty'] = df['novelty']
    y = df['d'].values
    X = np.column_stack([np.ones(len(y)), X.values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # novelty coefficient = last
    nov_coef = beta[-1]
    resid = y - X @ beta
    se = np.sqrt(np.sum(resid ** 2) / (len(y) - X.shape[1]))
    # crude t-stat for novelty coefficient (ignoring clustering — report as descriptive)
    print(f'  within-endpoint novelty slope: {nov_coef:+.5f} (raw)')
    print(f'  (clustered inference to be applied in full analysis)')

    # endpoint size x novelty interaction: correlate endpoint size with within-ep novelty slope
    slopes = {}
    for ep in df['endpoint'].unique():
        sub = df[df['endpoint'] == ep]
        if len(sub) < 50:
            continue
        sl, ic, r, p, se_ = stats.linregress(sub['novelty'], sub['d'])
        slopes[ep] = sl
    sdf = pd.DataFrame({'endpoint': list(slopes), 'slope': list(slopes.values())})
    sdf['n_mol'] = sdf['endpoint'].map(df.groupby('endpoint').size())
    print('  per-endpoint novelty slopes:')
    print(sdf.round(5).sort_values('slope'))
    rho, p = stats.spearmanr(sdf['n_mol'], sdf['slope'])
    print(f'  Spearman(endpoint size, novelty slope) = {rho:.3f} (p={p:.3f})')


def protocol_interaction(random_df, scaffold_df):
    print('\n=== C0: protocol interaction ===')
    agg_r = random_df.groupby('endpoint').d.mean()
    agg_s = scaffold_df.groupby('endpoint').d.mean()
    common = agg_r.index.intersection(agg_s.index)
    d_r = agg_r[common]
    d_s = agg_s[common]
    delta_r = d_r.mean()
    delta_s = d_s.mean()
    diff = delta_s - delta_r
    # paired per-endpoint protocol difference
    paired = pd.DataFrame({'random': d_r, 'scaffold': d_s})
    t, p = stats.ttest_rel(paired['random'], paired['scaffold'])
    print(f'  mean delta(random) = {delta_r:+.4f} | mean delta(scaffold) = {delta_s:+.4f}')
    print(f'  protocol difference = {diff:+.4f} (paired t={t:.2f}, p={p:.3f}, n={len(common)})')
    print('  per-endpoint:')
    print(paired.round(4))


def unique_molecules(df, tag):
    print(f'\n=== {tag}: unique molecule hygiene ===')
    print(f'  total rows: {len(df)}')
    print(f'  unique canonical molecules: {df["mol"].nunique()}')
    print(f'  unique molecules per endpoint: {df.groupby("endpoint")["mol"].nunique().to_dict()}')


def main():
    r = pd.read_parquet(f'{OUT}/b3_main_di.parquet')
    unique_molecules(r, 'random(R1b)')
    try:
        s = pd.read_parquet(f'{OUT}/b3_scaffold_di.parquet')
        unique_molecules(s, 'scaffold')
        protocol_interaction(r, s)
        within_endpoint_analysis(s, 'scaffold')
    except FileNotFoundError:
        print('\nscaffold supplement not ready yet (run b3_supplement.py first)')
        within_endpoint_analysis(r, 'random(R1b)')


if __name__ == '__main__':
    main()
