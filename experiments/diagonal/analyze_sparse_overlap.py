import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(base_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(base_dir, 'experiment_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Could not find experiment_results.csv at {csv_path}')
    df = pd.read_csv(csv_path)
    # Normalize column names if needed
    return df


def compute_final_val(df: pd.DataFrame) -> pd.DataFrame:
    # Build a unified final validation metric column
    # Prefer single-task final_val_mse; otherwise fall back to task 2, then task 1
    candidates = [
        'final_val_mse',
        'final_val_mse_task2',
        'final_val_mse_task1',
    ]
    present = [c for c in candidates if c in df.columns]
    if not present:
        raise ValueError('No final validation loss columns found in experiment_results.csv')
    df = df.copy()
    df['final_val'] = np.nan
    for c in present:
        df['final_val'] = df['final_val'].fillna(df[c])
    return df


def aggregate_by_n_train(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate across seeds/repetitions: mean, std, count, std_err per n_train2
    if 'n_train2' not in df.columns:
        raise ValueError('n_train2 column missing in experiment_results.csv')
    grouped = df.groupby('n_train2')['final_val']
    agg = grouped.agg(['mean', 'std', 'count']).reset_index()
    agg.rename(columns={'mean': 'final_val_mean', 'std': 'final_val_std', 'count': 'count'}, inplace=True)
    agg['final_val_std_err'] = agg['final_val_std'] / np.sqrt(agg['count'].replace(0, np.nan))
    return agg


def aggregate_by_seed_then_n_train(df: pd.DataFrame, normalize: bool = False, baseline_n: int = 16) -> pd.DataFrame:
    # Builds per-seed curves, optional normalization by seed baseline, then aggregates across seeds
    required = ['seed', 'n_train2', 'final_val']
    for col in required:
        if col not in df.columns:
            raise ValueError(f'Missing required column {col} in experiment_results.csv')
    df = df.copy()
    # Per-seed baseline normalization if requested
    if normalize:
        def _normalize_seed(g):
            g = g.copy()
            try:
                base = g.loc[g['n_train2'].astype(float) == baseline_n, 'final_val'].iloc[0]
            except IndexError:
                return pd.DataFrame(columns=g.columns)  # drop if no baseline
            if not np.isfinite(base) or base <= 0:
                return pd.DataFrame(columns=g.columns)
            g['final_val'] = g['final_val'] / base
            return g
        df = df.groupby('seed', group_keys=False).apply(_normalize_seed)
        if df.empty:
            return pd.DataFrame(columns=['n_train2', 'final_val_mean', 'final_val_std', 'count', 'final_val_std_err'])
    # Now aggregate across seeds at each n_train2
    return aggregate_by_n_train(df)


def plot_groups(df: pd.DataFrame, out_dir: str) -> None:
    # Identify initialization setups
    required_cols = ['init_method', 'lmda', 'c']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column {col} in experiment_results.csv')

    # Optionally include scaling if present
    has_scaling = 'scaling' in df.columns

    # Derive overlap_bool from numeric overlap
    if 'overlap' in df.columns:
        df = df.copy()
        df['overlap_bool'] = np.where(df['overlap'].astype(float) > 0, 'yes', 'no')

    # Compute global axes limits across all groups (log-log)
    x_all = df['n_train2'].values.astype(float)
    y_all = df['final_val'].values.astype(float)
    # guard: remove non-positive for log scale
    x_all = x_all[x_all > 0]
    y_all = y_all[y_all > 0]
    if len(x_all) == 0 or len(y_all) == 0:
        raise ValueError('Non-positive values encountered; cannot create log-log plots')
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()

    # Build combined figures with a 2x4 grid per (c, scaling):
    # Row 0 (active_dim_2 == 40): [simple, w_scaling=a] [simple, w_scaling=b] [complex,lmda=0] [complex,lmda=-10e-05]
    # Row 1 (active_dim_2 == 5):  [simple, w_scaling=a] [simple, w_scaling=b] [complex,lmda=0] [complex,lmda=-10e-05]
    # Group base by (c, scaling) so axes are shared and comparable
    base_group_cols = ['c'] + (['scaling'] if has_scaling else [])
    for base_vals, bdf in df.groupby(base_group_cols):
        if 'active_dim_2' not in bdf.columns:
            raise ValueError('Missing active_dim_2 in experiment_results.csv')
        if 'w_scaling' not in bdf.columns:
            raise ValueError('Missing w_scaling in experiment_results.csv')
        fig, axes = plt.subplots(2, 6, figsize=(28, 8), squeeze=False)

        # Define panel specs: (row, col, title, mask_func)
        def mask_simple(g):
            return (g['init_method'] == 'simple')
        def mask_complex0(g):
            return (g['init_method'] == 'complex') & (g['lmda'].astype(str).isin(['0', '0.0', '0.0000000000']))
        def mask_complexN(g):
            return (g['init_method'] == 'complex') & (g['lmda'].astype(str).isin(['-0.00001', '-0.0000100000', '-10e-05', '-1e-05']))

        # Determine up to two w_scaling values to display
        ws_uniqs = sorted(bdf['w_scaling'].dropna().unique())
        ws1 = ws_uniqs[0] if len(ws_uniqs) > 0 else None
        ws2 = ws_uniqs[1] if len(ws_uniqs) > 1 else None

        panel_defs = []
        # Row 0: active_dim_2 == 40
        panel_defs.append((0, 0, f'simple, w_scaling={ws1} (ad2=40)', lambda g: mask_simple(g) & (g['active_dim_2'] == 40) & (g['w_scaling'] == ws1)))
        panel_defs.append((0, 1, f'simple, w_scaling={ws2} (ad2=40)', lambda g: mask_simple(g) & (g['active_dim_2'] == 40) & (g['w_scaling'] == ws2)))
        panel_defs.append((0, 2, f'complex, lmda=0, w_scaling={ws1} (ad2=40)', lambda g: mask_complex0(g) & (g['active_dim_2'] == 40) & (g['w_scaling'] == ws1)))
        panel_defs.append((0, 3, f'complex, lmda=0, w_scaling={ws2} (ad2=40)', lambda g: mask_complex0(g) & (g['active_dim_2'] == 40) & (g['w_scaling'] == ws2)))
        panel_defs.append((0, 4, f'complex, lmda=-1e-5, w_scaling={ws1} (ad2=40)', lambda g: mask_complexN(g) & (g['active_dim_2'] == 40) & (g['w_scaling'] == ws1)))
        panel_defs.append((0, 5, f'complex, lmda=-1e-5, w_scaling={ws2} (ad2=40)', lambda g: mask_complexN(g) & (g['active_dim_2'] == 40) & (g['w_scaling'] == ws2)))
        # Row 1: active_dim_2 == 5
        panel_defs.append((1, 0, f'simple, w_scaling={ws1} (ad2=5)', lambda g: mask_simple(g) & (g['active_dim_2'] == 5) & (g['w_scaling'] == ws1)))
        panel_defs.append((1, 1, f'simple, w_scaling={ws2} (ad2=5)', lambda g: mask_simple(g) & (g['active_dim_2'] == 5) & (g['w_scaling'] == ws2)))
        panel_defs.append((1, 2, f'complex, lmda=0, w_scaling={ws1} (ad2=5)', lambda g: mask_complex0(g) & (g['active_dim_2'] == 5) & (g['w_scaling'] == ws1)))
        panel_defs.append((1, 3, f'complex, lmda=0, w_scaling={ws2} (ad2=5)', lambda g: mask_complex0(g) & (g['active_dim_2'] == 5) & (g['w_scaling'] == ws2)))
        panel_defs.append((1, 4, f'complex, lmda=-1e-5, w_scaling={ws1} (ad2=5)', lambda g: mask_complexN(g) & (g['active_dim_2'] == 5) & (g['w_scaling'] == ws1)))
        panel_defs.append((1, 5, f'complex, lmda=-1e-5, w_scaling={ws2} (ad2=5)', lambda g: mask_complexN(g) & (g['active_dim_2'] == 5) & (g['w_scaling'] == ws2)))

        for (r, cax, title, mask_fn) in panel_defs:
            ax = axes[r, cax]
            sub = bdf[mask_fn(bdf)]
            # Split by overlap yes/no and plot both
            if 'overlap_bool' in sub.columns and not sub.empty:
                subgroups = list(sub.groupby('overlap_bool'))
            else:
                subgroups = [('all', sub)]
            ax.set_xscale('log')
            ax.set_yscale('log')
            plotted_any = False
            for name, sg in subgroups:
                if sg.empty:
                    continue
                agg = aggregate_by_seed_then_n_train(sg, normalize=True, baseline_n=16)
                if agg.empty:
                    continue
                label = f'overlap={name}' if name != 'all' else 'final val (mean)'
                ax.plot(agg['n_train2'], agg['final_val_mean'], marker='o', linestyle='-', label=label)
                if 'final_val_std_err' in agg.columns and not np.isnan(agg['final_val_std_err']).all():
                    y0 = np.maximum(agg['final_val_mean'] - agg['final_val_std_err'], 1e-20)
                    y1 = agg['final_val_mean'] + agg['final_val_std_err']
                    ax.fill_between(agg['n_train2'], y0, y1, alpha=0.15)
                plotted_any = True
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('n_train2')
            if cax == 0:
                ax.set_ylabel('Final validation MSE')
            ax.set_title(title)
            if plotted_any:
                ax.legend(loc='best')

        # Title and save
        if has_scaling:
            c, scaling = base_vals if isinstance(base_vals, tuple) else (bdf['c'].iloc[0], bdf['scaling'].iloc[0])
            suptitle = f'c={c}, scaling={scaling}'
            fname = f'analysis_2x4__c={c}__scaling={scaling}.png'
        else:
            c = base_vals if not isinstance(base_vals, tuple) else base_vals[0]
            suptitle = f'c={c}'
            fname = f'analysis_2x4__c={c}.png'
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        # Also produce a 2x4 grid where rows are overlap (no/yes) and curves are active_dim_2=5 vs 40
        fig2, axes2 = plt.subplots(2, 6, figsize=(28, 8), squeeze=False)
        # panel columns same as above
        panel_defs2 = [
            (0, 0, f'simple, w_scaling={ws1} (overlap=no)', lambda g: mask_simple(g) & (g['overlap_bool'] == 'no') & (g['w_scaling'] == ws1)),
            (0, 1, f'simple, w_scaling={ws2} (overlap=no)', lambda g: mask_simple(g) & (g['overlap_bool'] == 'no') & (g['w_scaling'] == ws2)),
            (0, 2, f'complex, lmda=0, w_scaling={ws1} (overlap=no)', lambda g: mask_complex0(g) & (g['overlap_bool'] == 'no') & (g['w_scaling'] == ws1)),
            (0, 3, f'complex, lmda=0, w_scaling={ws2} (overlap=no)', lambda g: mask_complex0(g) & (g['overlap_bool'] == 'no') & (g['w_scaling'] == ws2)),
            (0, 4, f'complex, lmda=-1e-5, w_scaling={ws1} (overlap=no)', lambda g: mask_complexN(g) & (g['overlap_bool'] == 'no') & (g['w_scaling'] == ws1)),
            (0, 5, f'complex, lmda=-1e-5, w_scaling={ws2} (overlap=no)', lambda g: mask_complexN(g) & (g['overlap_bool'] == 'no') & (g['w_scaling'] == ws2)),
            (1, 0, f'simple, w_scaling={ws1} (overlap=yes)', lambda g: mask_simple(g) & (g['overlap_bool'] == 'yes') & (g['w_scaling'] == ws1)),
            (1, 1, f'simple, w_scaling={ws2} (overlap=yes)', lambda g: mask_simple(g) & (g['overlap_bool'] == 'yes') & (g['w_scaling'] == ws2)),
            (1, 2, f'complex, lmda=0, w_scaling={ws1} (overlap=yes)', lambda g: mask_complex0(g) & (g['overlap_bool'] == 'yes') & (g['w_scaling'] == ws1)),
            (1, 3, f'complex, lmda=0, w_scaling={ws2} (overlap=yes)', lambda g: mask_complex0(g) & (g['overlap_bool'] == 'yes') & (g['w_scaling'] == ws2)),
            (1, 4, f'complex, lmda=-1e-5, w_scaling={ws1} (overlap=yes)', lambda g: mask_complexN(g) & (g['overlap_bool'] == 'yes') & (g['w_scaling'] == ws1)),
            (1, 5, f'complex, lmda=-1e-5, w_scaling={ws2} (overlap=yes)', lambda g: mask_complexN(g) & (g['overlap_bool'] == 'yes') & (g['w_scaling'] == ws2)),
        ]
        for (r, cax, title, mask_fn) in panel_defs2:
            ax = axes2[r, cax]
            sub = bdf[mask_fn(bdf)]
            ax.set_xscale('log')
            ax.set_yscale('log')
            plotted_any = False
            # plot two curves for active_dim_2 = 5 and 40
            for ad2 in [5, 40]:
                sg = sub[sub['active_dim_2'] == ad2]
                if sg.empty:
                    continue
                agg = aggregate_by_seed_then_n_train(sg, normalize=True, baseline_n=16)
                if agg.empty:
                    continue
                ax.plot(agg['n_train2'], agg['final_val_mean'], marker='o', linestyle='-', label=f'active_dim_2={ad2}')
                if 'final_val_std_err' in agg.columns and not np.isnan(agg['final_val_std_err']).all():
                    y0 = np.maximum(agg['final_val_mean'] - agg['final_val_std_err'], 1e-20)
                    y1 = agg['final_val_mean'] + agg['final_val_std_err']
                    ax.fill_between(agg['n_train2'], y0, y1, alpha=0.15)
                plotted_any = True
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('n_train2')
            if cax == 0:
                ax.set_ylabel('Final validation MSE')
            ax.set_title(title)
            if plotted_any:
                ax.legend(loc='best')
        # Save second figure
        if has_scaling:
            c, scaling = base_vals if isinstance(base_vals, tuple) else (bdf['c'].iloc[0], bdf['scaling'].iloc[0])
            suptitle2 = f'c={c}, scaling={scaling} (rows: overlap=no/yes; curves: active_dim_2=5,40)'
            fname2 = f'analysis_2x4_by_overlap__c={c}__scaling={scaling}.png'
        else:
            c = base_vals if not isinstance(base_vals, tuple) else base_vals[0]
            suptitle2 = f'c={c} (rows: overlap=no/yes; curves: active_dim_2=5,40)'
            fname2 = f'analysis_2x4_by_overlap__c={c}.png'
        fig2.suptitle(suptitle2)
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path2 = os.path.join(out_dir, fname2)
        fig2.savefig(out_path2, dpi=200)
        plt.close(fig2)

        # Also produce raw (unnormalized) versions of both figures
        # First: 2x6 by active_dim_2 (raw values)
        fig3, axes3 = plt.subplots(2, 6, figsize=(28, 8), squeeze=False)
        for (r, cax, title, mask_fn) in panel_defs:
            ax = axes3[r, cax]
            sub = bdf[mask_fn(bdf)]
            # Split by overlap yes/no and plot both
            if 'overlap_bool' in sub.columns and not sub.empty:
                subgroups = list(sub.groupby('overlap_bool'))
            else:
                subgroups = [('all', sub)]
            ax.set_xscale('log')
            ax.set_yscale('log')
            plotted_any = False
            for name, sg in subgroups:
                if sg.empty:
                    continue
                agg = aggregate_by_seed_then_n_train(sg, normalize=False)
                if agg.empty:
                    continue
                # Use raw values (no normalization)
                label = f'overlap={name}' if name != 'all' else 'final val (mean)'
                ax.plot(agg['n_train2'], agg['final_val_mean'], marker='o', linestyle='-', label=label)
                if 'final_val_std_err' in agg.columns and not np.isnan(agg['final_val_std_err']).all():
                    y0 = np.maximum(agg['final_val_mean'] - agg['final_val_std_err'], 1e-20)
                    y1 = agg['final_val_mean'] + agg['final_val_std_err']
                    ax.fill_between(agg['n_train2'], y0, y1, alpha=0.15)
                plotted_any = True
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('n_train2')
            if cax == 0:
                ax.set_ylabel('Final validation MSE (raw)')
            ax.set_title(title)
            if plotted_any:
                ax.legend(loc='best')
        # Save third figure
        if has_scaling:
            c, scaling = base_vals if isinstance(base_vals, tuple) else (bdf['c'].iloc[0], bdf['scaling'].iloc[0])
            suptitle3 = f'c={c}, scaling={scaling} (raw values)'
            fname3 = f'analysis_2x6_raw__c={c}__scaling={scaling}.png'
        else:
            c = base_vals if not isinstance(base_vals, tuple) else base_vals[0]
            suptitle3 = f'c={c} (raw values)'
            fname3 = f'analysis_2x6_raw__c={c}.png'
        fig3.suptitle(suptitle3)
        fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path3 = os.path.join(out_dir, fname3)
        fig3.savefig(out_path3, dpi=200)
        plt.close(fig3)

        # Second: 2x6 by overlap (raw values)
        fig4, axes4 = plt.subplots(2, 6, figsize=(28, 8), squeeze=False)
        for (r, cax, title, mask_fn) in panel_defs2:
            ax = axes4[r, cax]
            sub = bdf[mask_fn(bdf)]
            ax.set_xscale('log')
            ax.set_yscale('log')
            plotted_any = False
            # plot two curves for active_dim_2 = 5 and 40
            for ad2 in [5, 40]:
                sg = sub[sub['active_dim_2'] == ad2]
                if sg.empty:
                    continue
                agg = aggregate_by_seed_then_n_train(sg, normalize=False)
                if agg.empty:
                    continue
                # Use raw values (no normalization)
                ax.plot(agg['n_train2'], agg['final_val_mean'], marker='o', linestyle='-', label=f'active_dim_2={ad2}')
                if 'final_val_std_err' in agg.columns and not np.isnan(agg['final_val_std_err']).all():
                    y0 = np.maximum(agg['final_val_mean'] - agg['final_val_std_err'], 1e-20)
                    y1 = agg['final_val_mean'] + agg['final_val_std_err']
                    ax.fill_between(agg['n_train2'], y0, y1, alpha=0.15)
                plotted_any = True
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('n_train2')
            if cax == 0:
                ax.set_ylabel('Final validation MSE (raw)')
            ax.set_title(title)
            if plotted_any:
                ax.legend(loc='best')
        # Save fourth figure
        if has_scaling:
            c, scaling = base_vals if isinstance(base_vals, tuple) else (bdf['c'].iloc[0], bdf['scaling'].iloc[0])
            suptitle4 = f'c={c}, scaling={scaling} (raw values, rows: overlap=no/yes; curves: active_dim_2=5,40)'
            fname4 = f'analysis_2x6_raw_by_overlap__c={c}__scaling={scaling}.png'
        else:
            c = base_vals if not isinstance(base_vals, tuple) else base_vals[0]
            suptitle4 = f'c={c} (raw values, rows: overlap=no/yes; curves: active_dim_2=5,40)'
            fname4 = f'analysis_2x6_raw_by_overlap__c={c}.png'
        fig4.suptitle(suptitle4)
        fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path4 = os.path.join(out_dir, fname4)
        fig4.savefig(out_path4, dpi=200)
        plt.close(fig4)

        # New: For w_scaling=1.0, plot four curves (lmda in {0,-1e-5} x active_dim_2 in {5,40})
        # as two subplots: left overlap=no, right overlap=yes (raw values)
        ws_target = 1.0
        sub_ws = bdf[(bdf['w_scaling'] == ws_target) & (bdf['init_method'] == 'complex')].copy()
        if not sub_ws.empty:
            fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
            for col_idx, ov in enumerate(['no', 'yes']):
                ax = axes5[0, col_idx]
                sub = sub_ws[sub_ws['overlap_bool'] == ov]
                ax.set_xscale('log')
                ax.set_yscale('log')
                plotted_any = False
                for lmda_tag, lmda_vals in [('0', ['0','0.0','0.0000000000']), ('-1e-5', ['-0.00001','-0.0000100000','-10e-05','-1e-05'])]:
                    for ad2 in [5, 40]:
                        sg = sub[(sub['active_dim_2'] == ad2) & (sub['lmda'].astype(str).isin(lmda_vals))]
                        if sg.empty:
                            continue
                        agg = aggregate_by_seed_then_n_train(sg, normalize=False)
                        if agg.empty:
                            continue
                        ax.plot(agg['n_train2'], agg['final_val_mean'], marker='o', linestyle='-', label=f'lmda={lmda_tag}, ad2={ad2}')
                        if 'final_val_std_err' in agg.columns and not np.isnan(agg['final_val_std_err']).all():
                            y0 = np.maximum(agg['final_val_mean'] - agg['final_val_std_err'], 1e-20)
                            y1 = agg['final_val_mean'] + agg['final_val_std_err']
                            ax.fill_between(agg['n_train2'], y0, y1, alpha=0.15)
                        plotted_any = True
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('n_train2')
                if col_idx == 0:
                    ax.set_ylabel('Final validation MSE (raw)')
                ax.set_title(f'overlap={ov}, w_scaling={ws_target}')
                if plotted_any:
                    ax.legend(loc='best', fontsize=8)
            # Save fifth figure
            if has_scaling:
                c, scaling = base_vals if isinstance(base_vals, tuple) else (bdf['c'].iloc[0], bdf['scaling'].iloc[0])
                suptitle5 = f'c={c}, scaling={scaling} (complex, w_scaling=1.0, four curves)'
                fname5 = f'analysis_overlap_fourcurves_ws1__c={c}__scaling={scaling}.png'
            else:
                c = base_vals if not isinstance(base_vals, tuple) else base_vals[0]
                suptitle5 = f'c={c} (complex, w_scaling=1.0, four curves)'
                fname5 = f'analysis_overlap_fourcurves_ws1__c={c}.png'
            fig5.suptitle(suptitle5)
            fig5.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path5 = os.path.join(out_dir, fname5)
            fig5.savefig(out_path5, dpi=200)
            plt.close(fig5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data/diagonal/sparse_overlap', help='Directory containing experiment_results.csv')
    args = parser.parse_args()
    df = load_results(args.base_dir)
    df = compute_final_val(df)
    plot_groups(df, args.base_dir)


if __name__ == '__main__':
    main()


