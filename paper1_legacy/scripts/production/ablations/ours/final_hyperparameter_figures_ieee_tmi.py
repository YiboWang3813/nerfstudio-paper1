"""Render H1/H2/H3 from frozen P2 sensitivity results at IEEE column size."""

import csv
import hashlib
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'docs/production/final_paper_evidence_20260910'
OUT = ROOT / 'docs/production/final_hyperparameter_figures_20260916/ieee_tmi_final'
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE, ORANGE, GRAY, BLACK = '#1F77B4', '#FF7F0E', '#8C8C8C', '#000000'
DATASETS = [('synthetic_liver', 'Liver', BLUE, 'o', '-'),
            ('tus_rec_2024_qc', 'Arm', ORANGE, 's', '--')]
HEAT = LinearSegmentedColormap.from_list(
    'tmi_diverging', ['#2166AC', '#F7F7F7', '#B2182B'])
HEAT_NORM = TwoSlopeNorm(vmin=-.016, vcenter=0, vmax=.016)
PANELS = [
    ('sigma_ele', ['.8', '1', '1.5', '2'], '1', r'$\sigma_{\mathrm{ele}}$'),
    ('beta', ['.4', '.6', '.8', '1'], '.8', r'$\beta$'),
    ('lambda_ray', ['.2', '.35', '.55', '.75', '1'], '.55', r'$\lambda_{\mathrm{ray}}$'),
    ('Nq', ['1', '2', '3'], '3', r'$N_q$'),
    ('half_window', ['1', '2', '3', '4'], '2', 'half-window'),
]


def read_tsv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def write_tsv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def load():
    manifest = json.loads((SRC / 'audits/final_data_source_manifest.json').read_text())
    index = {r['row_id']: r for r in read_tsv(SRC / 'audits/run_sources.tsv')}
    paths = {'H1': SRC / 'ppp/rho_data.tsv',
             'H2': SRC / 'lgr/hyperparameter_data.tsv',
             'H3': SRC / 'cga/hyperparameter_data.tsv'}
    data = {}
    for name, path in paths.items():
        rel = path.relative_to(SRC).as_posix()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['package_file_sha256'][rel]
        rows = read_tsv(path)
        for row in rows:
            trace = index[row['row_id']]
            assert row['output_dir'] == trace['output_dir']
            assert row['execution_source_sha256'] == trace['execution_source_sha256']
            assert trace['validated_p2_source_sha256'] == manifest['P2_source_sha256']
            cfg = yaml.safe_load((SRC / trace['packaged_config']).read_text())
            sr = cfg['model']['slice_render']
            assert sr['local_conditioning_mode'] == 'fixed'
            assert sr['ray_attenuation_enabled'] and sr['ray_attenuation_formula'] == 'integral'
            assert cfg['engine']['iterations'] == 30000 and cfg['system']['seed'] == 42
            assert cfg['scene']['sampler']['num_init'] == 1000000
            assert cfg['scene']['loading']['pose_mode'] == ('canonical_sweep' if name == 'H1' else 'file')
        data[name] = rows
    assert [len(data[name]) for name in ('H1', 'H2', 'H3')] == [14, 40, 42]
    return data, paths, manifest['P2_source_sha256']


def style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams.update({
        'svg.fonttype': 'none',
        'mathtext.fontset': 'dejavusans', 'font.size': 7,
        'axes.labelsize': 7.8, 'xtick.labelsize': 6.7,
        'ytick.labelsize': 6.7, 'legend.fontsize': 6.8,
        'axes.linewidth': .65, 'lines.linewidth': 1.1,
        'lines.markersize': 3.2, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'legend.frameon': False})


def frame(ax, grid=True):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(.65)
    ax.tick_params(width=.55, length=2, pad=1.5, labelsize=6.7)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, color='#E5E5E5', linewidth=.4)


def delta_rows(rows, field, expected, default):
    plotted = []
    for dataset, name, *_ in DATASETS:
        subset = sorted((r for r in rows if r['dataset'] == dataset),
                        key=lambda r: Decimal(r[field]))
        assert [Decimal(r[field]) for r in subset] == [Decimal(v) for v in expected]
        base = next(r for r in subset if Decimal(r[field]) == Decimal(default))
        for row in subset:
            delta = Decimal(row['final_psnr']) - Decimal(base['final_psnr'])
            assert abs(delta - Decimal(row['delta_psnr'])) < Decimal('1e-12')
            plotted.append(dict(dataset=name, parameter=field, value=row[field],
                                default_value=default, row_id=row['row_id'],
                                psnr=row['final_psnr'],
                                delta_psnr=str(delta if delta else Decimal(0)),
                                lpips=row['final_lpips'],
                                final_G=str(int(Decimal(row['final_num_gaussians']))),
                                is_default=str(row is base)))
    return plotted


def save(fig, name, axes=None, panel_ids=None, row_groups=None, column_groups=None,
         exclude_axes=None):
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / name
    if axes is not None:
        require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=panel_ids,
            row_groups=row_groups, column_groups=column_groups,
            exclude_axes=exclude_axes or [], json_out=str(base)+'.alignment.json',
            overlay_svg=str(base)+'.alignment.svg', tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.width and box.height and (
                box.x0 < -1 or box.y0 < -1 or
                box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
                outside.append(artist.get_text())
    assert not outside, (name, outside)
    assert fig.get_figwidth() == 3.5
    fig.savefig(str(base)+'.pdf', facecolor='white',
                metadata={'Creator': 'ULGSplat frozen sensitivity data',
                          'CreationDate': None})
    fig.savefig(str(base)+'.png', dpi=600, facecolor='white',
                transparent=False)
    fig.savefig(str(base)+'.svg', facecolor='white')
    plt.close(fig)


def h1(rows):
    values = ['.6', '.75', '.9', '1', '1.1', '1.25', '1.5']
    plotted = delta_rows(rows, 'rho', values, '1')
    fig, ax = plt.subplots(figsize=(3.5, 1.4))
    fig.subplots_adjust(left=.145, right=.985, bottom=.31, top=.82)
    for _, name, color, marker, linestyle in DATASETS:
        sub = [r for r in plotted if r['dataset'] == name]
        ax.plot([float(r['value']) for r in sub],
                [float(r['delta_psnr']) for r in sub],
                color=color, marker=marker, linestyle=linestyle,
                markerfacecolor=color, markeredgecolor=color, label=name)
    ax.axvline(1, color=GRAY, linewidth=.65, linestyle=(0, (3, 2)))
    ax.axhline(0, color=GRAY, linewidth=.55)
    ax.plot(1, 0, 'ks', markersize=4.7, zorder=8)
    ax.plot(1, 0, 'ko', markersize=2.8, zorder=9)
    ax.set_xticks([float(v) for v in values])
    ax.set_xticklabels(['.60', '.75', '.90', '1.00', '1.10', '1.25', '1.50'])
    ax.set_xlim(.56, 1.54)
    ax.set_ylim(-.55, .42)
    ax.set_yticks([-.5, -.25, 0, .25])
    ax.set_xlabel(r'Spacing scale $\rho$', labelpad=1)
    ax.set_ylabel(r'$\Delta$PSNR (dB)', labelpad=2)
    frame(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',
               bbox_to_anchor=(.985, .995), ncol=2, columnspacing=.85,
               handlelength=1.5, handletextpad=.35)
    save(fig, 'H1_AFGM')
    write_tsv(OUT / 'H1_AFGM_plotted_data.tsv', plotted)


def h2(rows):
    fig, axes = plt.subplots(5, 1, figsize=(3.5, 3.8))
    fig.subplots_adjust(left=.205, right=.985, bottom=.07, top=.93, hspace=.52)
    all_plotted = []
    for ax, (panel, values, default, label) in zip(axes, PANELS):
        plotted = delta_rows([r for r in rows if r['panel'] == panel],
                             'point', values, default)
        all_plotted.extend(dict(r, panel=panel) for r in plotted)
        for _, name, color, marker, linestyle in DATASETS:
            sub = [r for r in plotted if r['dataset'] == name]
            ax.plot([float(r['value']) for r in sub],
                    [float(r['delta_psnr']) for r in sub],
                    color=color, marker=marker, linestyle=linestyle,
                    markerfacecolor=color, markeredgecolor=color, label=name)
        ax.axvline(float(default), color=GRAY, linewidth=.55,
                   linestyle=(0, (3, 2)))
        ax.axhline(0, color=GRAY, linewidth=.45)
        ax.plot(float(default), 0, 'ks', markersize=4.6, zorder=8)
        ax.plot(float(default), 0, 'ko', markersize=2.7, zorder=9)
        xs = [float(v) for v in values]
        ax.set_xticks(xs)
        ax.set_xticklabels([format(Decimal(v), 'g') for v in values])
        margin = .09 * (max(xs)-min(xs))
        ax.set_xlim(min(xs)-margin, max(xs)+margin)
        ys = [float(r['delta_psnr']) for r in plotted]
        span = max(max(ys)-min(ys), .02)
        upper_margin = .75 if panel == 'Nq' else .55
        ax.set_ylim(min(min(ys), 0)-.17*span,
                    max(max(ys), 0)+upper_margin*span)
        if panel == 'sigma_ele':
            ax.set_yticks([-.8, -.4, 0])
        elif panel == 'beta':
            ax.set_yticks([-.15, 0])
        elif panel == 'Nq':
            ax.set_yticks([-.8, -.4, 0])
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        frame(ax, grid=False)
        ax.grid(True, axis='y', color='#E5E5E5', linewidth=.4)
        ax.text(.025, .94 if panel == 'Nq' else .89,
                label, transform=ax.transAxes,
                ha='left', va='top', fontsize=8, color=BLACK)
    fig.text(.035, .5, r'$\Delta$PSNR (dB)', rotation=90,
             rotation_mode='anchor',
             ha='center', va='center', fontsize=7.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.61, .996),
               ncol=2, columnspacing=1, handlelength=1.6, handletextpad=.35)
    save(fig, 'H2_PLGR', axes=list(axes),
         panel_ids=['a', 'b', 'c', 'd', 'e'],
         column_groups=[['a', 'b', 'c', 'd', 'e']])
    write_tsv(OUT / 'H2_PLGR_plotted_data.tsv', all_plotted)


def text_color(delta):
    red, green, blue = HEAT(HEAT_NORM(delta))[:3]
    luminance = .299*red + .587*green + .114*blue
    return 'white' if luminance < .48 else BLACK


def population_k(value):
    return str((Decimal(value)/1000).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def h3(rows):
    assert {r['panel'] for r in rows} == {'qd_by_B', 'qp_by_qA', 'tau_alpha'}
    fig = plt.figure(figsize=(3.5, 2.1))
    left = fig.add_axes([.16, .405, .225, .50])
    right = fig.add_axes([.575, .405, .225, .50])
    strip = fig.add_axes([.16, .145, .64, .13])
    cax = fig.add_axes([.865, .405, .022, .50])
    plotted = []
    spec = [
        ('qd_by_B', left, ['.96', '.98', '.99'], ['.001', '.005', '.05'],
         ('.98', '.05'), r'$q_d$', r'$B$'),
        ('qp_by_qA', right, ['.05', '.10', '.15'], ['.70', '.80', '.90'],
         ('.10', '.80'), r'$q_p$', r'$q_A$'),
    ]
    mesh = None
    for panel, ax, ylabels, xlabels, default, ylabel, xlabel in spec:
        panel_rows = [r for r in rows if r['panel'] == panel]
        assert len(panel_rows) == 18
        ys, xs = [Decimal(v) for v in ylabels], [Decimal(v) for v in xlabels]
        matrix = np.empty((6, 3), dtype=float)
        pop = np.empty((6, 3), dtype=object)
        for ds_index, (dataset, name, *_) in enumerate(DATASETS):
            subset = [r for r in panel_rows if r['dataset'] == dataset]
            assert len(subset) == 9
            base = next(r for r in subset if r['is_default'] == 'True')
            for r in subset:
                y, x = [Decimal(v) for v in r['point'].split('_')]
                assert y in ys and x in xs
                ri, ci = 3*ds_index+ys.index(y), xs.index(x)
                delta = Decimal(r['final_psnr'])-Decimal(base['final_psnr'])
                assert abs(delta-Decimal(r['delta_psnr'])) < Decimal('1e-12')
                matrix[ri, ci] = float(delta)
                pop[ri, ci] = population_k(r['final_num_gaussians'])
                plotted.append(dict(dataset=name, panel=panel, y_value=str(y),
                                    x_value=str(x), row_id=r['row_id'],
                                    delta_psnr=str(delta if delta else Decimal(0)),
                                    final_G=r['final_num_gaussians'],
                                    annotation_G_k=pop[ri, ci],
                                    is_default=str(r is base)))
        mesh = ax.pcolormesh(np.arange(4)-.5, np.arange(7)-.5, matrix,
                             cmap=HEAT, norm=HEAT_NORM, shading='flat',
                             edgecolors='#F2F2F2', linewidth=.45,
                             rasterized=False)
        for ri in range(6):
            for ci in range(3):
                ax.text(ci, ri, pop[ri, ci], ha='center', va='center',
                        fontsize=6.6, color=text_color(matrix[ri, ci]))
        for ds_index in (0, 1):
            ri = 3*ds_index+ys.index(Decimal(default[0]))
            ci = xs.index(Decimal(default[1]))
            ax.add_patch(Rectangle((ci-.47, ri-.47), .94, .94,
                                   fill=False, edgecolor=BLACK, linewidth=.7))
        ax.axhline(2.5, color='#B9B9B9', linewidth=.8)
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(5.5, -.5)
        ax.set_xticks(range(3))
        if panel == 'qd_by_B':
            ax.set_xticklabels(['1e-3', '5e-3', '5e-2'])
        else:
            ax.set_xticklabels(['.70', '.80', '.90'])
        ax.set_yticks(range(6))
        ax.set_yticklabels([('{} {}'.format('L' if i < 3 else 'A',
                                            format(ys[i % 3], 'g')))
                            for i in range(6)])
        ax.set_xlabel(xlabel, labelpad=.8)
        ax.set_ylabel(ylabel, labelpad=.8)
        frame(ax, grid=False)

    tau_rows = [r for r in rows if r['panel'] == 'tau_alpha']
    assert len(tau_rows) == 6
    taus = [Decimal('.00005'), Decimal('.0001'), Decimal('.0005')]
    tau_matrix = np.empty((2, 3), dtype=float)
    tau_pop = np.empty((2, 3), dtype=object)
    for ds_index, (dataset, name, *_) in enumerate(DATASETS):
        subset = [r for r in tau_rows if r['dataset'] == dataset]
        base = next(r for r in subset if r['is_default'] == 'True')
        for r in subset:
            x = Decimal(r['point'])
            assert x in taus
            ci = taus.index(x)
            delta = Decimal(r['final_psnr'])-Decimal(base['final_psnr'])
            assert abs(delta-Decimal(r['delta_psnr'])) < Decimal('1e-12')
            tau_matrix[ds_index, ci] = float(delta)
            tau_pop[ds_index, ci] = population_k(r['final_num_gaussians'])
            plotted.append(dict(dataset=name, panel='tau_alpha', y_value='',
                                x_value=str(x), row_id=r['row_id'],
                                delta_psnr=str(delta if delta else Decimal(0)),
                                final_G=r['final_num_gaussians'],
                                annotation_G_k=tau_pop[ds_index, ci],
                                is_default=str(r is base)))
    strip.pcolormesh(np.arange(4)-.5, np.arange(3)-.5, tau_matrix,
                     cmap=HEAT, norm=HEAT_NORM, shading='flat',
                     edgecolors='#F2F2F2', linewidth=.45, rasterized=False)
    for ri in range(2):
        for ci in range(3):
            strip.text(ci, ri, tau_pop[ri, ci], ha='center', va='center',
                       fontsize=6.6, color=text_color(tau_matrix[ri, ci]))
        strip.add_patch(Rectangle((.53, ri-.47), .94, .94,
                                  fill=False, edgecolor=BLACK, linewidth=.7))
    strip.set_xlim(-.5, 2.5)
    strip.set_ylim(1.5, -.5)
    strip.set_xticks(range(3))
    strip.set_xticklabels(['5e-5', '1e-4', '5e-4'])
    strip.set_yticks([0, 1])
    strip.set_yticklabels(['Liver', 'Arm'])
    strip.set_xlabel(r'$\tau_\alpha$', labelpad=3)
    frame(strip, grid=False)
    cb = fig.colorbar(mesh, cax=cax, ticks=[-.016, 0, .016])
    cb.solids.set_rasterized(False)
    cb.ax.set_yticklabels(['−.016', '0', '+.016'])
    cb.ax.tick_params(labelsize=6.5, width=.5, length=1.8, pad=1)
    cb.ax.yaxis.set_label_position('left')
    cb.set_label('ΔPSNR vs default\n(dB)', fontsize=6.7, labelpad=0)
    cb.outline.set_linewidth(.55)
    fig.text(.16, .975, 'L: Liver    A: Arm', fontsize=6.7,
             ha='left', va='top')
    fig.text(.785, .975, 'text: #G (k)', fontsize=6.7,
             ha='right', va='top')
    save(fig, 'H3_CGAS', axes=[left, right, strip],
         panel_ids=['a', 'b', 'c'], row_groups=[['a', 'b']],
         exclude_axes=[cax])
    write_tsv(OUT / 'H3_CGAS_plotted_data.tsv', plotted)


def main():
    style()
    data, paths, p2 = load()
    h1(data['H1'])
    h2(data['H2'])
    h3(data['H3'])
    (OUT / 'source_manifest.json').write_text(json.dumps({
        'P2_source_sha256': p2,
        'source_files': {
            key: {'path': str(path),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in paths.items()},
        'figure_sizes_inches': {'H1_AFGM': [3.5, 1.4],
                                'H2_PLGR': [3.5, 3.8],
                                'H3_CGAS': [3.5, 2.1]},
        'no_training': True,
    }, indent=2)+'\n')
    print('Rendered H1/H2/H3 in {}'.format(OUT))


if __name__ == '__main__':
    main()
