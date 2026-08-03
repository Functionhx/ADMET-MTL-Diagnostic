"""
B3 figures v3 (mentor round-2 redesign, P0-7):
Fig 1: Experimental framework (three-layer, shows 70/10/20 + same-model calibration
       + GIN + controls + confirmatory null result)
Fig 2: Two-panel merged forest: (A) per-endpoint Delta random/scaffold with CIs;
       (B) per-endpoint Gamma_e with CIs
Fig 3: Novelty slope forest (8 endpoints, per-protocol slopes + CI)
Fig 4: Paired calibration plot (per-endpoint raw -> calibrated lines + Gamma)
Fig 5: Mechanism estimation plot (STL 0 / MTL / shuffled / pooled with split points)
Output: paper/figures/fig1_framework.pdf, fig2_forest.pdf, fig3_gamma.pdf,
        fig4_novelty.pdf, fig5_calibration.pdf, fig6_mechanism.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = '/home/as/vllm/cell/idea-stage/b3_v2_out'
FIG = '/home/as/vllm/cell/paper/figures'
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7})


def fig1_framework():
    """Three-layer horizontal framework: data -> allocation -> models -> four outputs."""
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.axis('off')
    b_data = dict(boxstyle='round,pad=0.4', facecolor='#F0F7F0', edgecolor='#55A868', lw=1.2)
    b_alloc = dict(boxstyle='round,pad=0.4', facecolor='#EAF0F6', edgecolor='#4C72B0', lw=1.2)
    b_model = dict(boxstyle='round,pad=0.4', facecolor='#FDF3E7', edgecolor='#DD8452', lw=1.2)
    b_out = dict(boxstyle='round,pad=0.4', facecolor='#F7F2F7', edgecolor='#8172B3', lw=1.2)

    def draw(x, y, text, b, fs=7.5, w=1.0):
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, bbox=b)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=9, color='#555555', lw=1.1))

    # layer 1: data
    draw(0.5, 0.86, 'Eight binary ADMET endpoints (TDC)\ncanonical-SMILES round-trip, dedup rule', b_data, w=1.9)
    arrow(0.5, 0.70, 0.5, 0.58)
    # layer 2: global allocation
    draw(0.5, 0.46, 'Global allocation: 70% train / 10% calibration / 20% test\n'
                    'random (molecule)  vs  scaffold-grouped\n'
                    '5 split instances x 3 seeds (no cross-task identity exposure)', b_alloc, w=1.9)
    arrow(0.5, 0.30, 0.5, 0.18)
    # layer 3: models
    draw(0.5, 0.08, 'Hard-sharing MTL (task-balanced)\nvs architecture-matched STL', b_model, w=1.3)
    arrow(0.28, -0.06, 0.17, -0.18); arrow(0.50, -0.06, 0.50, -0.18); arrow(0.72, -0.06, 0.83, -0.18)
    # layer 4: four outputs
    draw(0.17, -0.30, 'Positive protocol\ncontrast (Gamma = +0.081)', b_out, w=0.62)
    draw(0.50, -0.30, 'Same-model\ncalibration: raw advantage\nattenuated after calibration', b_out, w=0.62)
    draw(0.83, -0.30, 'Mechanism\ncontrols\n(label-permuted)', b_out, w=0.62)
    arrow(0.17, -0.42, 0.17, -0.54)
    draw(0.17, -0.62, 'Matched-scale GIN\nsensitivity (Gamma = +0.026)', b_out, w=0.62)
    arrow(0.83, -0.42, 0.83, -0.54)
    draw(0.83, -0.62, 'Novelty: flat\n(3-instance reversal spurious)', b_out, w=0.62)
    ax.set_xlim(0, 1); ax.set_ylim(-0.78, 1.02)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig1_framework.pdf')
    plt.close(fig)
    print('fig1_framework saved')


def fig2_merged(r, s, ci_csv, ge_csv):
    """Merged two-panel: (A) per-endpoint Delta with CIs; (B) Gamma_e forest with CIs."""
    ci = pd.read_csv(ci_csv) if ci_csv else None
    ge = pd.read_csv(ge_csv) if ge_csv else None
    ep_disp = {'hERG': 'hERG', 'AMES': 'AMES', 'BBB_Martins': 'BBB Martins',
               'Pgp_Broccatelli': 'P-gp Broccatelli', 'CYP2C9_Veith': 'CYP2C9 Veith',
               'CYP2D6_Veith': 'CYP2D6 Veith', 'CYP3A4_Veith': 'CYP3A4 Veith',
               'Bioavailability_Ma': 'Bioavailability Ma'}
    eps = [e for e in ['hERG', 'AMES', 'BBB_Martins', 'Pgp_Broccatelli',
                       'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
                       'Bioavailability_Ma'] if e in r.endpoint.unique()]
    ytick_labs = [ep_disp[e] for e in eps]
    y = np.arange(len(eps))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True,
                             gridspec_kw={'width_ratios': [1.6, 1]})
    axA, axB = axes
    for i, (ep, yi) in enumerate(zip(eps, y)):
        m_r = r[r.endpoint == ep].d.mean()
        m_s = s[s.endpoint == ep].d.mean()
        lo_r, hi_r = None, None
        if ci is not None:
            row = ci[(ci.endpoint == ep) & (ci.protocol == 'random')]
            if len(row): lo_r, hi_r = row.ci_lo.iloc[0], row.ci_hi.iloc[0]
        lo_s, hi_s = None, None
        if ci is not None:
            row = ci[(ci.endpoint == ep) & (ci.protocol == 'scaffold')]
            if len(row): lo_s, hi_s = row.ci_lo.iloc[0], row.ci_hi.iloc[0]
        if lo_r is not None:
            axA.errorbar(m_r, yi + 0.18, xerr=[[m_r - lo_r], [hi_r - m_r]], fmt='o', ms=4,
                         color='#4C72B0', capsize=2, label='random' if i == 0 else None)
        else:
            axA.plot(m_r, yi + 0.18, 'o', ms=4, color='#4C72B0', label='random' if i == 0 else None)
        if lo_s is not None:
            axA.errorbar(m_s, yi - 0.18, xerr=[[m_s - lo_s], [hi_s - m_s]], fmt='s', ms=4,
                         color='#DD8452', capsize=2, label='scaffold' if i == 0 else None)
        else:
            axA.plot(m_s, yi - 0.18, 's', ms=4, color='#DD8452', label='scaffold' if i == 0 else None)
        G = m_s - m_r
        gl, gh = None, None
        if ge is not None:
            row = ge[ge.endpoint == ep]
            if len(row): gl, gh = row.ci_lo.iloc[0], row.ci_hi.iloc[0]
        if gl is not None:
            axB.errorbar(G, yi, xerr=[[G - gl], [gh - G]], fmt='o', ms=4,
                         color='#C44E52', capsize=2)
        else:
            axB.plot(G, yi, 'o', ms=5, color='#C44E52')
    axA.axvline(0, color='black', lw=0.8)
    axB.axvline(0, color='black', lw=0.8)
    Gmean = (s.groupby('endpoint').d.mean() - r.groupby('endpoint').d.mean()).mean()
    axB.axvline(Gmean, color='#C44E52', ls='--', lw=1,
                label=f'mean Γ = {Gmean:+.3f}')
    axA.set_yticks(y); axA.set_yticklabels(ytick_labs, fontsize=6.5)
    axA.set_xlabel(r'$\Delta$NLL = NLL$_{STL}$ − NLL$_{MTL}$')
    axA.set_title('(A) Per-endpoint contrast', fontsize=8.5)
    axA.legend(frameon=False, fontsize=6.5)
    axB.set_xlabel('Γ$_e$ = Δ$_{scaf}$ − Δ$_{rand}$')
    axB.set_title('(B) Protocol interaction (forest)', fontsize=8.5)
    axB.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig2_forest.pdf')
    plt.close(fig)
    print('fig2_forest (merged, forest CI) saved')


def fig3_novelty_forest(r, s):
    """Novelty slopes per endpoint (8 endpoints), per protocol + CI."""
    rows = []
    for df, tag in [(r, 'random'), (s, 'scaffold')]:
        for ep, g in df.groupby('endpoint'):
            g = g.copy()
            g['bin'] = pd.qcut(g['novelty'], 6, labels=False, duplicates='drop')
            b = g.groupby('bin').d.mean()
            x = g.groupby('bin').novelty.median()
            slope = np.polyfit(x.values, b.values, 1)[0]
            # bootstrap slope CI over (inst, seed) units
            slopes_b = []
            rng = np.random.RandomState(hash(ep + tag) % 2**31)
            units = g.groupby(['inst', 'seed']).size()
            for _ in range(300):
                idx = rng.choice(len(units), len(units), replace=True)
                sub = g[g.groupby(['inst', 'seed']).ngroup().isin(
                    g.groupby(['inst', 'seed']).ngroup().unique()[idx])]
                sb = sub.copy(); sb['bin'] = pd.qcut(sb['novelty'], 6, labels=False, duplicates='drop')
                bb = sb.groupby('bin').d.mean(); xx = sb.groupby('bin').novelty.median()
                if len(bb) >= 3:
                    slopes_b.append(np.polyfit(xx.values, bb.values, 1)[0])
            if slopes_b:
                ci_lo, ci_hi = np.percentile(slopes_b, [2.5, 97.5])
            else:
                ci_lo = ci_hi = np.nan
            rows.append({'endpoint': ep, 'protocol': tag, 'slope': slope,
                         'ci_lo': ci_lo, 'ci_hi': ci_hi})
    d = pd.DataFrame(rows)
    eps = r.endpoint.unique()
    y_r = np.arange(len(eps))[::-1] + 0.15
    y_s = np.arange(len(eps))[::-1] - 0.15
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, ep in enumerate(eps):
        rr = d[(d.endpoint == ep) & (d.protocol == 'random')]
        ss = d[(d.endpoint == ep) & (d.protocol == 'scaffold')]
        ax.errorbar(rr.slope.iloc[0], y_r[i],
                    xerr=[[rr.slope.iloc[0] - rr.ci_lo.iloc[0]], [rr.ci_hi.iloc[0] - rr.slope.iloc[0]]],
                    fmt='o', ms=4, color='#4C72B0', capsize=2,
                    label='random' if i == 0 else None)
        ax.errorbar(ss.slope.iloc[0], y_s[i],
                    xerr=[[ss.slope.iloc[0] - ss.ci_lo.iloc[0]], [ss.ci_hi.iloc[0] - ss.slope.iloc[0]]],
                    fmt='s', ms=4, color='#DD8452', capsize=2,
                    label='scaffold' if i == 0 else None)
    ax.axvline(0, color='black', lw=0.8)
    ep_disp = {'hERG': 'hERG', 'AMES': 'AMES', 'BBB_Martins': 'BBB Martins',
               'Pgp_Broccatelli': 'P-gp Broccatelli', 'CYP2C9_Veith': 'CYP2C9 Veith',
               'CYP2D6_Veith': 'CYP2D6 Veith', 'CYP3A4_Veith': 'CYP3A4 Veith',
               'Bioavailability_Ma': 'Bioavailability Ma'}
    ax.set_yticks(np.arange(len(eps)))
    ax.set_yticklabels([ep_disp.get(e, e) for e in eps], fontsize=6.5)
    ax.set_xlabel('within-endpoint novelty slope (binned ΔNLL per unit novelty)')
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title('Endpoint-level novelty slopes under both protocols', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig4_novelty.pdf')
    plt.close(fig)
    print('fig4_novelty (slope forest) saved')


def fig4_paired_calibration(r, s):
    """Paired plot: per-endpoint raw -> calibrated lines + Gamma raw->cal."""
    dr_raw = r.groupby('endpoint').d.mean()
    dr_cal = r.groupby('endpoint').d_cal.mean()
    ds_raw = s.groupby('endpoint').d.mean()
    ds_cal = s.groupby('endpoint').d_cal.mean()
    eps = dr_raw.index
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, pairs, xlab, title in [
        (axes[0], [(dr_raw, dr_cal)], 'random: raw → calibrated', 'Random allocation'),
        (axes[1], [(ds_raw, ds_cal)], 'scaffold: raw → calibrated', 'Scaffold allocation'),
        (axes[2], [(ds_raw - dr_raw, ds_cal - dr_cal)], 'Γ: raw → calibrated', 'Protocol contrast')]:
        for i, (raw, cal) in enumerate(pairs):
            for ep in eps:
                ax.plot([i, i + 1], [raw[ep], cal[ep]], color='#999999', lw=0.8, alpha=0.7)
                ax.plot(i, raw[ep], 'o', ms=4, color='#4C72B0')
                ax.plot(i + 1, cal[ep], 'o', ms=4, color='#C44E52')
            ax.plot([i, i + 1], [raw.mean(), cal.mean()], color='black', lw=2,
                    label='endpoint mean')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['raw', 'calibrated'], fontsize=7)
        ax.set_xlabel(xlab, fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel(r'$\Delta$NLL = NLL$_{STL}$ − NLL$_{MTL}$')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig5_calibration.pdf')
    plt.close(fig)
    print('fig5_calibration (paired) saved')


def fig5_mechanism():
    """Mechanism estimation plot: STL 0 / MTL / shuffled / pooled with split points."""
    import os as _os
    c = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_controls_v2_out/control_v2_di.parquet')
    v1 = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_controls_out/control_di.parquet')
    c = c.merge(v1[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl', 'p_mtl']],
                on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')
    configs = [('Standard MTL', 'p_mtl'), ('Label-permuted MTL', 'p_mtl_shuff'),
               ('Pooled pretrain + STL', 'p_mtl_pooled')]
    # pooled 来自 v1（p_mtl_pooled 不在 merge 后）——重新读取
    c1 = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_controls_out/control_di.parquet')
    def nll(y, p):
        eps = 1e-7
        return -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = [0, 1, 2]
    rng = np.random.RandomState(0)
    # MTL + shuffled 从 merged (target-specific)；pooled 从 v1
    pooled = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_controls_out/control_di.parquet')
    for x, (tag, col, src) in zip(xs, [('Standard MTL', 'p_mtl', c),
                                       ('Label-permuted MTL', 'p_mtl_shuff', c),
                                       ('Pooled pretrain + STL', 'p_mtl_pooled', pooled)]):
        d_all = []
        for (inst, seed), g in src.groupby(['inst', 'seed']):
            d = g.apply(lambda r_: nll(r_['y'], r_['p_stl']).mean() -
                        nll(r_['y'], r_[col]).mean(), axis=1)
            d_all.append(d.mean())
        d_all = np.array(d_all)
        jit = rng.uniform(-0.08, 0.08, len(d_all))
        ax.scatter(x + jit, d_all, s=8, color='#999999', alpha=0.7, zorder=2)
        ax.errorbar(x, d_all.mean(), yerr=1.96 * d_all.std() / np.sqrt(len(d_all)),
                    fmt='o', ms=8, color='#C44E52', capsize=3, zorder=3)
        ax.annotate(f'{d_all.mean():+.3f}', (x, d_all.mean()),
                    xytext=(0, 8), textcoords='offset points', ha='center',
                    fontsize=7.5, color='#C44E52')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xticks(xs)
    ax.set_xticklabels(['Standard\nMTL', 'Label-permuted\nMTL', 'Pooled pretrain\n+ STL'], fontsize=7)
    ax.set_ylabel(r'$\Delta$NLL vs STL (positive favors config)')
    ax.set_title('Fully permuted auxiliary labels leave the gain intact', fontsize=8.5)
    ax.set_ylim(-0.15, 0.42)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig6_mechanism.pdf')
    plt.close(fig)
    print('fig6_mechanism saved')


if __name__ == '__main__':
    r = pd.read_parquet(f'{OUT}/b3_v2_random_di.parquet')
    s = pd.read_parquet(f'{OUT}/b3_v2_scaffold_di.parquet')
    ci_csv = f'{OUT}/r2c/per_endpoint_ci.csv'
    ge_csv = f'{OUT}/r2c/gamma_e_ci.csv'
    fig1_framework()
    fig2_merged(r, s, ci_csv, ge_csv)
    fig3_novelty_forest(r, s)
    fig4_paired_calibration(r, s)
    fig5_mechanism()
    print('all v3 figures saved')
