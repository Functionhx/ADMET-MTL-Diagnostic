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
plt.rcParams.update({'font.size': 9.5, 'axes.labelsize': 9.5, 'legend.fontsize': 8.5})


def fig1_framework():
    """Horizontal causal chain: raw finding -> checkpoint -> calibration -> controls -> GIN,
    with measured box extents so no text is clipped and arrows connect actual box edges."""
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.axis('off')
    b_main = dict(boxstyle='round,pad=0.3', facecolor='#EAF0F6', edgecolor='#4C72B0', lw=1.2)
    b_find = dict(boxstyle='round,pad=0.3', facecolor='#FDF3E7', edgecolor='#DD8452', lw=1.2)
    b_diag = dict(boxstyle='round,pad=0.3', facecolor='#F0F7F0', edgecolor='#55A868', lw=1.2)
    b_sec = dict(boxstyle='round,pad=0.3', facecolor='#F2F2F2', edgecolor='#AAAAAA', lw=0.9)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    ext = {}

    def add(x, y, text, style, fs=8.5, key=None):
        t_ = ax.text(0, 0, text, ha='center', va='center', fontsize=fs, bbox=style)
        bb = t_.get_window_extent(renderer=rend)
        (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        w, h = x1 - x0, y1 - y0
        t_.set_position((x, y))
        ext[key or text] = (x - w / 2, x + w / 2, y - h / 2, y + h / 2)
        return w, h

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=11, color='#555555', lw=1.2))

    # Main chain (top row, left to right)
    add(0.13, 0.72, 'Leakage-controlled\nMTL vs STL\n(8 TDC endpoints, 70/10/20)', b_main, key='A')
    add(0.50, 0.72, 'Raw finding:\nMTL gain larger\nunder scaffold split', b_find, key='B')
    # Right column diagnostics (top to bottom)
    add(0.87, 0.72, 'Checkpoint rule:\nvalidation-selected reverses\nabsolute contrast,\nmean Gamma positive', b_diag, key='C')
    add(0.87, 0.34, 'Calibration:\nper-model T attenuates\nproper-score differences', b_diag, key='D')
    add(0.87, -0.04, 'Label permutation +\nSTL-8x: gain survives\npermutation, not budget', b_diag, key='E')
    add(0.50, -0.28, 'Cross-architecture:\nGIN directionally supportive', b_diag, fs=8, key='F')
    # Secondary analyses (grey, bottom-left)
    add(0.13, 0.12, 'Secondary analyses:\nnovelty (flat) / downsampling /\npartition geometry / top-k decision', b_sec, fs=7.5, key='G')
    add(0.13, -0.12, 'Dev analyses excluded;\nplan frozen (protocol/)', b_sec, fs=7.5, key='H')

    # Arrows between measured box edges
    e = lambda k: ext[k]
    arrow(e('A')[1] + 0.012, 0.72, e('B')[0] - 0.012, 0.72)
    arrow(e('B')[1] + 0.012, 0.72, e('C')[0] - 0.012, 0.72)
    arrow(0.87, e('C')[3] - 0.012, 0.87, e('D')[2] + 0.012)
    arrow(0.87, e('D')[3] - 0.012, 0.87, e('E')[2] + 0.012)
    arrow(e('E')[1] - 0.015, e('E')[3], e('F')[0] + 0.015, e('F')[2])

    allx = [ext[k][0] for k in ext] + [ext[k][1] for k in ext]
    ally = [ext[k][2] for k in ext] + [ext[k][3] for k in ext]
    ax.set_xlim(min(allx) - 0.02, max(allx) + 0.02)
    ax.set_ylim(min(ally) - 0.05, max(ally) + 0.05)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig1_framework.pdf')
    plt.close(fig)
    print('fig1_framework (measured layout) saved')


def fig2_merged(r, s, ci_csv, ge_csv):
    """Merged two-panel: (A) per-endpoint Delta with molecule-cluster CIs;
    (B) per-endpoint Gamma_e forest with split-instance bootstrap CIs
    (seeds averaged within each split; the split, not the split-seed run,
    is the independent uncertainty unit because seeds share the test set)."""
    ci = pd.read_csv(ci_csv) if ci_csv else None
    rng = np.random.RandomState(7)
    ep_disp = {'hERG': 'hERG', 'AMES': 'AMES', 'BBB_Martins': 'BBB Martins',
               'Pgp_Broccatelli': 'P-gp Broccatelli', 'CYP2C9_Veith': 'CYP2C9 Veith',
               'CYP2D6_Veith': 'CYP2D6 Veith', 'CYP3A4_Veith': 'CYP3A4 Veith',
               'Bioavailability_Ma': 'Bioavailability Ma'}
    eps = [e for e in ['hERG', 'AMES', 'BBB_Martins', 'Pgp_Broccatelli',
                       'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
                       'Bioavailability_Ma'] if e in r.endpoint.unique()]
    ytick_labs = [ep_disp[e] for e in eps]
    y = np.arange(len(eps))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True,
                             gridspec_kw={'width_ratios': [1.6, 1]})
    axA, axB = axes
    for i, (ep, yi) in enumerate(zip(eps, y)):
        sub_r = r[r.endpoint == ep]; sub_s = s[s.endpoint == ep]
        m_r = sub_r.d.mean(); m_s = sub_s.d.mean()
        lo_r = hi_r = lo_s = hi_s = None
        if ci is not None:
            row = ci[(ci.endpoint == ep) & (ci.protocol == 'random')]
            if len(row): lo_r, hi_r = row.ci_lo.iloc[0], row.ci_hi.iloc[0]
            row = ci[(ci.endpoint == ep) & (ci.protocol == 'scaffold')]
            if len(row): lo_s, hi_s = row.ci_lo.iloc[0], row.ci_hi.iloc[0]
        if lo_r is not None:
            axA.errorbar(m_r, yi + 0.18, xerr=[[m_r - lo_r], [hi_r - m_r]], fmt='o', ms=5,
                         color='#4C72B0', capsize=2, label='random' if i == 0 else None)
        else:
            axA.plot(m_r, yi + 0.18, 'o', ms=5, color='#4C72B0', label='random' if i == 0 else None)
        if lo_s is not None:
            axA.errorbar(m_s, yi - 0.18, xerr=[[m_s - lo_s], [hi_s - m_s]], fmt='s', ms=5,
                         color='#DD8452', capsize=2, label='scaffold' if i == 0 else None)
        else:
            axA.plot(m_s, yi - 0.18, 's', ms=5, color='#DD8452', label='scaffold' if i == 0 else None)
        # Panel B: split-level Gamma_e (seeds averaged within split), split-instance bootstrap
        g_r = sub_r.groupby('inst').d.mean()
        g_s = sub_s.groupby('inst').d.mean()
        g_e = (g_s - g_r).values
        G = g_e.mean()
        boots = []
        for _ in range(3000):
            idx = rng.choice(len(g_e), len(g_e), replace=True)
            boots.append(g_e[idx].mean())
        gl, gh = np.percentile(boots, [2.5, 97.5])
        axB.errorbar(G, yi, xerr=[[G - gl], [gh - G]], fmt='o', ms=5,
                     color='#C44E52', capsize=2)
    axA.axvline(0, color='black', lw=0.8)
    axB.axvline(0, color='black', lw=0.8)
    Gmean = (s.groupby('endpoint').d.mean() - r.groupby('endpoint').d.mean()).mean()
    axB.axvline(Gmean, color='#C44E52', ls='--', lw=1,
                label=f'mean Γ = {Gmean:+.3f}')
    axA.set_yticks(y); axA.set_yticklabels(ytick_labs, fontsize=8)
    axA.set_xlabel(r'$\Delta$NLL = NLL$_{STL}$ − NLL$_{MTL}$')
    axA.set_title('(A) Per-endpoint contrast', fontsize=10)
    axA.legend(frameon=False, fontsize=8)
    axB.set_xlabel('Γ$_e$ = Δ$_{scaf}$ − Δ$_{rand}$')
    axB.set_title('(B) Protocol interaction (forest)', fontsize=10)
    axB.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig2_forest.pdf')
    plt.close(fig)
    print('fig2_forest (split-instance forest CI) saved')


def fig3_novelty_forest(r, s):
    """Novelty slopes per endpoint (8 endpoints), per protocol + CI.
    Slope bootstrap resamples split instances (seeds averaged within split)."""
    rows = []
    for df, tag in [(r, 'random'), (s, 'scaffold')]:
        for ep, g in df.groupby('endpoint'):
            g = g.copy()
            g['bin'] = pd.qcut(g['novelty'], 6, labels=False, duplicates='drop')
            b = g.groupby('bin').d.mean()
            x = g.groupby('bin').novelty.median()
            slope = np.polyfit(x.values, b.values, 1)[0]
            # bootstrap slope CI over split instances (units = splits)
            slopes_b = []
            rng = np.random.RandomState(hash(ep + tag) % 2**31)
            insts = g.inst.unique()
            for _ in range(1000):
                chosen = rng.choice(insts, len(insts), replace=True)
                sub = g[g.inst.isin(chosen)]
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
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for i, ep in enumerate(eps):
        rr = d[(d.endpoint == ep) & (d.protocol == 'random')]
        ss = d[(d.endpoint == ep) & (d.protocol == 'scaffold')]
        ax.errorbar(rr.slope.iloc[0], y_r[i],
                    xerr=[[rr.slope.iloc[0] - rr.ci_lo.iloc[0]], [rr.ci_hi.iloc[0] - rr.slope.iloc[0]]],
                    fmt='o', ms=5, color='#4C72B0', capsize=2,
                    label='random' if i == 0 else None)
        ax.errorbar(ss.slope.iloc[0], y_s[i],
                    xerr=[[ss.slope.iloc[0] - ss.ci_lo.iloc[0]], [ss.ci_hi.iloc[0] - ss.slope.iloc[0]]],
                    fmt='s', ms=5, color='#DD8452', capsize=2,
                    label='scaffold' if i == 0 else None)
    ax.axvline(0, color='black', lw=0.8)
    ep_disp = {'hERG': 'hERG', 'AMES': 'AMES', 'BBB_Martins': 'BBB Martins',
               'Pgp_Broccatelli': 'P-gp Broccatelli', 'CYP2C9_Veith': 'CYP2C9 Veith',
               'CYP2D6_Veith': 'CYP2D6 Veith', 'CYP3A4_Veith': 'CYP3A4 Veith',
               'Bioavailability_Ma': 'Bioavailability Ma'}
    ax.set_yticks(np.arange(len(eps)))
    ax.set_yticklabels([ep_disp.get(e, e) for e in eps], fontsize=8)
    ax.set_xlabel('within-endpoint novelty slope (binned ΔNLL per unit novelty)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Endpoint-level novelty slopes under both protocols', fontsize=10)
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
        ax.set_xticks([0, 1]); ax.set_xticklabels(['raw', 'calibrated'], fontsize=8.5)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r'$\Delta$NLL = NLL$_{STL}$ − NLL$_{MTL}$')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig5_calibration.pdf')
    plt.close(fig)
    print('fig5_calibration (paired) saved')


def fig5_mechanism():
    """Two-panel mechanism figure:
    (A) random-protocol controls (STL-8x / MTL / permuted / pooled vs STL),
        split-level points (seeds averaged within split), no run-level CIs;
    (B) protocol interaction under real vs permuted labels, 3 paired split-level
        values (no bootstrap: seeds share the test set and only 3 splits exist)."""
    v2 = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_v2_out/b3_v2_random_di.parquet')
    stl8x = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_stl8x_out/stl8x_di.parquet')
    stl8x = stl8x[stl8x.seed.isin(v2.seed.unique())]
    stl8x = stl8x.merge(v2[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl']],
                        on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')
    pooled = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_controls_v2_out/pooled_v2_di.parquet')
    pooled = pooled.merge(v2[['inst', 'seed', 'endpoint', 'mol', 'p_stl']],
                         on=['inst', 'seed', 'endpoint', 'mol'], how='left')
    c = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_controls_v2_out/control_v2_di.parquet')
    c = c.merge(v2[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl', 'p_mtl']],
                on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')

    def nll(y, p):
        eps = 1e-7
        return -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))

    def split_means(src, col, stl_col='p_stl'):
        """Endpoint-macro contrast per split (seeds averaged within split)."""
        vals = []
        for inst, g in src.groupby('inst'):
            gg = g.dropna(subset=[col, stl_col])
            if len(gg) == 0:
                continue
            d = gg.apply(lambda r_: nll(r_['y'], r_[stl_col]).mean() -
                         nll(r_['y'], r_[col]).mean(), axis=1)
            vals.append(d.mean())
        return np.array(vals)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={'width_ratios': [1.4, 1]})
    axA, axB = axes

    xs = [0, 1, 2, 3]
    groups = [('STL-8x', stl8x, 'p_stl8x'),
              ('Standard\nMTL', c, 'p_mtl'),
              ('Label-\npermuted', c, 'p_mtl_shuff'),
              ('Pooled\n+ STL', pooled, 'p_mtl_pooled')]
    for x, (tag, src, col) in zip(xs, groups):
        d_all = split_means(src, col)
        if len(d_all) == 0:
            continue
        axA.scatter(np.full(len(d_all), x), d_all, s=26, color='#4C72B0',
                    alpha=0.75, zorder=2, edgecolors='white', linewidths=0.5)
        axA.plot(x, d_all.mean(), 'o', ms=9, color='#C44E52', zorder=3)
        axA.annotate(f'{d_all.mean():+.3f}', (x, d_all.mean()),
                     xytext=(0, 9), textcoords='offset points', ha='center',
                     fontsize=8.5, color='#C44E52')
    axA.axhline(0, color='black', lw=0.8, ls='--')
    axA.set_xticks(xs)
    axA.set_xticklabels([g[0] for g in groups], fontsize=8)
    axA.set_ylabel(r'$\Delta$NLL vs STL (positive favors config)')
    axA.set_title('(A) Random-protocol controls (split-level)', fontsize=10)
    axA.set_ylim(-0.25, 0.48)

    r3 = v2[v2.inst.isin([0, 1, 2])]
    c2 = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_controls_v2_out/control_v2_di.parquet')
    mr = c2.merge(r3[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl', 'p_mtl']],
                  on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')
    s3 = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_v2_out/b3_v2_scaffold_di.parquet')
    s3 = s3[s3.inst.isin([0, 1, 2])]
    sp = pd.read_parquet('/home/as/vllm/cell/idea-stage/b3_controls_scaffold_out/scaffold_permuted_di.parquet')
    ms = sp.merge(s3[['inst', 'seed', 'endpoint', 'mol', 'y', 'p_stl', 'p_mtl']],
                  on=['inst', 'seed', 'endpoint', 'mol', 'y'], how='left')

    def gamma_per_split(df, model_col):
        vals = []
        for inst, g in df.groupby('inst'):
            ep_vals = [nll(gg.y, gg.p_stl).mean() - nll(gg.y, gg[model_col]).mean()
                       for ep, gg in g.groupby('endpoint')]
            vals.append(np.mean(ep_vals))
        return np.array(vals)

    g_real = gamma_per_split(ms, 'p_mtl')
    g_perm = gamma_per_split(ms, 'p_mtl_shuff')
    # paired split-level values for real and permuted labels
    for k in range(len(g_real)):
        axB.plot([0, 1], [g_real[k], g_perm[k]], color='#999999', lw=0.9, alpha=0.8, zorder=1)
    axB.scatter(np.zeros(len(g_real)), g_real, s=28, color='#4C72B0', zorder=3, edgecolors='white', linewidths=0.5)
    axB.scatter(np.ones(len(g_perm)), g_perm, s=28, color='#4C72B0', zorder=3, edgecolors='white', linewidths=0.5)
    for x, v in [(0, g_real.mean()), (1, g_perm.mean())]:
        axB.plot(x, v, 'o', ms=9, color='#C44E52', zorder=4)
        axB.annotate(f'{v:+.3f}', (x, v), xytext=(0, 9), textcoords='offset points',
                     ha='center', fontsize=8.5, color='#C44E52')
    axB.axhline(0, color='black', lw=0.8, ls='--')
    axB.set_xticks([0, 1])
    axB.set_xticklabels(['Real labels', 'Permuted labels'], fontsize=8)
    axB.set_ylabel(r'$\Gamma$ (scaffold $-$ random)')
    axB.set_title('(B) Interaction survives permutation (3 splits)', fontsize=10)
    axB.set_ylim(0, 0.22)

    fig.tight_layout()
    fig.savefig(f'{FIG}/fig6_mechanism.pdf')
    plt.close(fig)
    print('fig6_mechanism (split-level) saved')


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
