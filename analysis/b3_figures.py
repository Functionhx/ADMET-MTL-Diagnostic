"""
B3 figure generation (runs after R1c + S1-S4 complete).
Fig 1: protocol x endpoint-scale interaction panel
Fig 2: ΔNLL vs log endpoint size (with downsampling arrows)
Fig 3: within-endpoint ΔNLL vs novelty (endpoint-controlled residuals)
Table 1: per-endpoint ΔNLL/CI/AUC/AUPRC/Brier/unique molecules
Output: paper/figures/*.pdf + table CSVs
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = 'b3_main_out'
FIGDIR = '/home/as/vllm/cell/paper/figures'
import os
os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8})


def brier_decomp(y, p):
    brier = np.mean((p - y) ** 2)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    rel = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        rel += m.sum() / len(y) * (np.mean(p[m]) - np.mean(y[m])) ** 2
    return brier, rel


def fig1_protocol_scale(random_df, scaffold_df):
    """Fig 1: ΔNLL by protocol x endpoint-scale stratum."""
    agg_r = random_df.groupby('endpoint').d.mean()
    agg_s = scaffold_df.groupby('endpoint').d.mean()
    sizes = random_df.groupby('endpoint').size()
    common = agg_r.index.intersection(agg_s.index)
    d_r, d_s = agg_r[common], agg_s[common]
    n = sizes[common]
    small = n < 6000
    fig, ax = plt.subplots(figsize=(4.5, 3))
    x = np.arange(len(common))
    w = 0.35
    ax.bar(x - w/2, d_r.values, w, label='random protocol', color='#4C72B0')
    ax.bar(x + w/2, d_s.values, w, label='scaffold protocol', color='#DD8452')
    for i, (sm, dr, ds) in enumerate(zip(small.values, d_r.values, d_s.values)):
        if sm:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels([e.split('_')[0] for e in common], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(r'$\Delta$NLL (MTL$-$STL)')
    ax.axhline(0, color='black', lw=0.8)
    ax.legend(frameon=False)
    ax.set_title('Protocol × endpoint scale')
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/fig1_protocol_scale.pdf')
    plt.close(fig)
    print('fig1 saved')


def fig2_scale(df, ds_df=None):
    """Fig 2: ΔNLL vs log endpoint size."""
    agg = df.groupby('endpoint').agg(mean_d=('d', 'mean'), n=('d', 'size'),
                                     se=('d', 'std'))
    agg['se'] /= np.sqrt(agg['n'])
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.errorbar(np.log10(agg.n), agg.mean_d, yerr=1.96 * agg.se, fmt='o', capsize=3,
                color='#4C72B0', label='endpoints')
    for ep, r in agg.iterrows():
        ax.annotate(ep.split('_')[0], (np.log10(r.n), r.mean_d), fontsize=7, ha='center',
                    xytext=(0, 5), textcoords='offset points')
    if ds_df is not None:
        dagg = ds_df.groupby('endpoint').d.mean()
        # downsample points (CYP at n=2000): annotate arrows
        for ep in ['CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith']:
            if ep in dagg and ep in agg:
                ax.plot(np.log10(2000), dagg[ep], 's', color='#C44E52')
                ax.annotate('', xy=(np.log10(2000), dagg[ep]),
                            xytext=(np.log10(agg.loc[ep, 'n']), agg.loc[ep, 'mean_d']),
                            arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1))
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('log10(endpoint training size)')
    ax.set_ylabel(r'$\Delta$NLL (MTL$-$STL)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/fig2_scale.pdf')
    plt.close(fig)
    print('fig2 saved')


def fig3_novelty(df):
    """Fig 3: within-endpoint binned ΔNLL vs novelty (endpoint-controlled)."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for ep in df.endpoint.unique()[:4]:  # show 4 endpoints
        sub = df[df.endpoint == ep]
        sub = sub.copy()
        sub['bin'] = pd.qcut(sub['novelty'], 6, labels=False, duplicates='drop')
        b = sub.groupby('bin').agg(m=('d', 'mean'), nov=('novelty', 'median'),
                                   se=('d', 'std'), n=('d', 'size'))
        b['se'] /= np.sqrt(b['n'])
        ax.errorbar(b.nov, b.m, yerr=1.96 * b.se, fmt='o-', capsize=2, ms=3,
                    label=ep.split('_')[0])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('nearest-train Tanimoto (novelty)')
    ax.set_ylabel(r'$\Delta$NLL (MTL$-$STL)')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/fig3_novelty.pdf')
    plt.close(fig)
    print('fig3 saved')


def table1(df):
    """Table 1: per-endpoint summary."""
    rows = []
    for ep in df.endpoint.unique():
        sub = df[df.endpoint == ep]
        y, ps, pm = sub.y.values, sub.p_stl.values, sub.p_mtl.values
        auc_s = roc_auc_score(y, ps) if len(np.unique(y)) > 1 else np.nan
        auc_m = roc_auc_score(y, pm) if len(np.unique(y)) > 1 else np.nan
        aupr_s = average_precision_score(y, ps)
        aupr_m = average_precision_score(y, pm)
        bs, rels = brier_decomp(y, ps)
        bm, relm = brier_decomp(y, pm)
        rows.append({'endpoint': ep, 'n_test': len(sub), 'n_unique': sub.mol.nunique(),
                     'dNLL': sub.d.mean(), 'dNLL_se': sub.d.std() / np.sqrt(len(sub)),
                     'AUC_stl': auc_s, 'AUC_mtl': auc_m,
                     'AUPRC_stl': aupr_s, 'AUPRC_mtl': aupr_m,
                     'Brier_stl': bs, 'Brier_mtl': bm,
                     'Rel_stl': rels, 'Rel_mtl': relm})
    t = pd.DataFrame(rows)
    t.to_csv(f'{OUT}/table1_summary.csv', index=False)
    print('table1 saved')
    print(t.round(4).to_string(index=False))


def main():
    r = pd.read_parquet(f'{OUT}/b3_main_di.parquet')
    table1(r)
    fig2_scale(r)
    fig3_novelty(r)
    try:
        s = pd.read_parquet(f'{OUT}/b3_scaffold_di.parquet')
        fig1_protocol_scale(r, s)
    except FileNotFoundError:
        print('scaffold data not ready; fig1 skipped')
    try:
        ds = pd.read_parquet(f'{OUT}/b3_scaffold_ds_di.parquet')
        fig2_scale(r, ds)
    except FileNotFoundError:
        print('downsample data not ready; fig2 arrows skipped')


if __name__ == '__main__':
    main()
