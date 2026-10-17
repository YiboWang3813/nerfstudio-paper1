"""H1/H2 with the user-specified boxed, hollow-marker IEEE visual grammar."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from final_hyperparameter_figures_ieee_tmi import (
    ROOT, SRC, PANELS, load, delta_rows, write_tsv)

OUT = ROOT / 'docs/production/final_hyperparameter_figures_20260916/ieee_tmi_endo_style'
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

LIVER = '#D95319'
ARM = '#2878B5'
GRID = '#D9D9D9'
REFERENCE = '#777777'
SPINE = '#222222'
DATASETS = [('Liver', LIVER, 's', '-'), ('Arm', ARM, 'o', '--')]


def active_font():
    """Use real Times New Roman when installed, otherwise its metric-compatible substitute."""
    try:
        font_manager.findfont(FontProperties(family='Times New Roman'),
                              fallback_to_default=False)
        return 'Times New Roman'
    except ValueError:
        return 'Liberation Serif'


FONT = active_font()


def style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [FONT]
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT,
        'mathtext.it': FONT + ':italic',
        'mathtext.bf': FONT + ':bold',
        'font.size': 7,
        'axes.labelsize': 7.8,
        'xtick.labelsize': 6.8,
        'ytick.labelsize': 6.8,
        'legend.fontsize': 6.8,
        'axes.linewidth': .8,
        'lines.linewidth': 1.25,
        'lines.markersize': 4.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'legend.frameon': False,
    })


def frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE)
        spine.set_linewidth(.8)
    ax.tick_params(width=.65, length=2, pad=1.5, labelsize=6.8,
                   color=SPINE, labelcolor=SPINE)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', color=GRID, linewidth=.5,
            linestyle=(0, (2.2, 2.2)))


def draw_pair(ax, plotted):
    for name, color, marker, linestyle in DATASETS:
        sub = [r for r in plotted if r['dataset'] == name]
        x = [float(r['value']) for r in sub]
        y = [float(r['delta_psnr']) for r in sub]
        # Pairwise path segments have exactly the same visible polyline while
        # keeping vector stroke bounds local to each interval.
        for i in range(len(x)-1):
            ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=1.25,
                    linestyle=linestyle, zorder=2)
        ax.plot(x, y, color=color, linestyle='None', marker=marker,
                markersize=4.0,
                markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=1.0, label=name)


def save(fig, name, axes=None):
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / name
    if axes is not None:
        require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=['a', 'b', 'c', 'd', 'e'],
            column_groups=[['a', 'b', 'c', 'd', 'e']],
            json_out=str(base)+'.alignment.json',
            overlay_svg=str(base)+'.alignment.svg',
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.width and box.height and (
                    box.x0 < -1 or box.y0 < -1 or
                    box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
                outside.append((artist.get_text(), tuple(round(v, 2) for v in box.bounds)))
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
    fig, ax = plt.subplots(figsize=(3.5, 1.0))
    fig.subplots_adjust(left=.12, right=.995, bottom=.32, top=.985)
    draw_pair(ax, plotted)
    ax.axvline(1, color=REFERENCE, linewidth=.75,
               linestyle=(0, (3, 2)), zorder=0)
    ax.set_xticks([float(v) for v in values])
    ax.set_xticklabels(['.60', '.75', '.90', '1.00', '1.10', '1.25', '1.50'])
    ax.set_xlim(.56, 1.54)
    ax.set_ylim(-.55, .42)
    ax.set_yticks([-.5, -.25, 0, .25])
    ax.set_xlabel(r'Spacing scale $\rho$', labelpad=2)
    ax.set_ylabel(r'$\Delta$PSNR (dB)', labelpad=0)
    frame(ax)
    ax.grid(False, axis='y')
    legend = ax.legend(loc='upper right', ncol=2, columnspacing=.7,
              handlelength=1.35, handletextpad=.35, borderaxespad=.35,
              frameon=True, facecolor='white', edgecolor=GRID,
              framealpha=1, fancybox=False, borderpad=.25)
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    legend_box = legend_box.expanded(1.06, 1.12)
    inv = ax.transData.inverted()
    legend_lo = inv.transform((legend_box.x0, legend_box.y0))
    legend_hi = inv.transform((legend_box.x1, legend_box.y1))
    lx0, ly0 = legend_lo
    lx1, ly1 = legend_hi
    grid_style = dict(color=GRID, linewidth=.5,
                      linestyle=(0, (2.2, 2.2)), zorder=0)
    for tick in [-.5, -.25, 0, .25]:
        stop = lx0 if ly0 <= tick <= ly1 else 1.54
        ax.hlines(tick, .56, stop, **grid_style)
    for tick in [float(v) for v in values]:
        if lx0 <= tick <= lx1:
            ax.vlines(tick, -.55, ly0, **grid_style)
        else:
            ax.vlines(tick, -.55, .42, **grid_style)
    save(fig, 'H1_AFGM')
    write_tsv(OUT / 'H1_AFGM_plotted_data.tsv', plotted)


def h2(rows):
    fig, axes = plt.subplots(5, 1, figsize=(3.5, 3.7))
    fig.subplots_adjust(left=.108, right=.995, bottom=.082, top=.98,
                        hspace=.66)
    plgr_datasets = [('Liver', '#2E7D32', '*', '-'),
                     ('Arm', '#7B3294', 'D', '--')]
    axis_specs = {
        'sigma_ele': ((.70, 2.10), [.75, 1, 1.25, 1.5, 1.75, 2],
                      (-.85, .55), [-.8, -.5, -.2, .1, .4], 'upper left'),
        'beta': ((.35, 1.05), [.4, .6, .8, 1],
                 (-.18, .24), [-.15, -.05, .05, .15], 'upper left'),
        'lambda_ray': ((.15, 1.05), [.2, .4, .6, .8, 1],
                       (-.065, .035), [-.06, -.03, 0, .03], 'upper left'),
        'Nq': ((.75, 3.25), [1, 2, 3],
               (-.83, .17), [-.8, -.5, -.2, .1], 'lower left'),
        'half_window': ((.70, 4.30), [1, 2, 3, 4],
                        (-.032, .018), [-.03, -.01, .01], 'upper left'),
    }
    method_labels = {
        'sigma_ele': r'Slice width $\sigma^{\mathrm{ele}}$',
        'beta': r'Composition weight $\beta$',
        'lambda_ray': r'Ray attenuation $\lambda_{\mathrm{ray}}$',
        'Nq': r'Quadrature order $N_q$',
        'half_window': r'Half-window $w$',
    }
    all_plotted = []
    for ax, (panel, values, default, label) in zip(axes, PANELS):
        plotted = delta_rows([r for r in rows if r['panel'] == panel],
                             'point', values, default)
        all_plotted.extend(dict(r, panel=panel) for r in plotted)
        x_spec, xticks, y_spec, yticks, legend_loc = axis_specs[panel]
        for name, color, marker, linestyle in plgr_datasets:
            sub = [r for r in plotted if r['dataset'] == name]
            ax.plot([float(r['value']) for r in sub],
                    [float(r['delta_psnr']) for r in sub],
                    color=color, linewidth=1.2, linestyle=linestyle,
                    marker=marker,
                    markersize=6.2 if name == 'Liver' else 3.5,
                    markerfacecolor='white',
                    markeredgewidth=1, markeredgecolor=color,
                    label=name, zorder=3)
        assert all(x_spec[0] < float(v) < x_spec[1] for v in values)
        assert all(y_spec[0] < float(r['delta_psnr']) < y_spec[1]
                   for r in plotted)
        ax.set_xlim(*x_spec)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format(t, 'g') for t in xticks])
        ax.set_ylim(*y_spec)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        frame(ax)
        ax.grid(False, axis='y')
        ax.axvline(float(default), color=REFERENCE, linewidth=.75,
                   linestyle=(0, (3, 2)), zorder=1)
        ax.set_ylabel(r'$\Delta$PSNR (dB)', labelpad=0, fontsize=7.2)
        ax.yaxis.set_label_coords(-.079, .5)
        ax.set_xlabel(method_labels[panel],
                      labelpad=1.2 if panel == 'sigma_ele' else .5,
                      fontsize=7.8)
        # Marker-only keys keep the legend line from crossing its own text.
        legend_handles = [
            Line2D([], [], linestyle='None', marker=marker,
                   markersize=6.2 if name == 'Liver' else 3.5,
                   markerfacecolor='white', markeredgecolor=color,
                   markeredgewidth=1)
            for name, color, marker, _ in plgr_datasets
        ]
        legend = ax.legend(legend_handles, ['Liver', 'Arm'],
                  loc=legend_loc, ncol=2,
                  columnspacing=.50, handlelength=.45, handletextpad=.30,
                  borderaxespad=.20 if legend_loc == 'upper left' else .50,
                  fontsize=6.1, frameon=True, facecolor='white',
                  edgecolor=GRID, framealpha=1, fancybox=False,
                  borderpad=.12)
        # Avoid only the legend's *actual* small rectangle, not a broad
        # left-hand strip. Grid strokes must not cross legend lettering.
        fig.canvas.draw()
        legend_box = legend.get_window_extent(fig.canvas.get_renderer())
        legend_box = legend_box.expanded(1.08, 1.14)
        inv = ax.transData.inverted()
        legend_lo = inv.transform((legend_box.x0, legend_box.y0))
        legend_hi = inv.transform((legend_box.x1, legend_box.y1))
        lx0, ly0 = legend_lo
        lx1, ly1 = legend_hi
        grid_style = dict(color=GRID, linewidth=.5,
                          linestyle=(0, (2.2, 2.2)), zorder=0)
        for tick in yticks:
            start = min(x_spec[1], lx1) if ly0 <= tick <= ly1 else x_spec[0]
            ax.hlines(tick, start, x_spec[1], **grid_style)
        for tick in xticks:
            if lx0 <= tick <= lx1:
                if ly0 > y_spec[0]:
                    ax.vlines(tick, y_spec[0], ly0, **grid_style)
                if ly1 < y_spec[1]:
                    ax.vlines(tick, ly1, y_spec[1], **grid_style)
            else:
                ax.vlines(tick, *y_spec, **grid_style)
    save(fig, 'H2_PLGR', axes=list(axes))
    write_tsv(OUT / 'H2_PLGR_plotted_data.tsv', all_plotted)


def main():
    style()
    data, paths, p2 = load()
    h1(data['H1'])
    h2(data['H2'])
    (OUT / 'source_manifest.json').write_text(json.dumps({
        'P2_source_sha256': p2,
        'requested_font': 'Times New Roman',
        'embedded_font_family': FONT,
        'source_files': {
            key: {'path': str(paths[key]),
                  'sha256': hashlib.sha256(paths[key].read_bytes()).hexdigest()}
            for key in ('H1', 'H2')},
        'figure_sizes_inches': {'H1_AFGM': [3.5, 1.0],
                                'H2_PLGR': [3.5, 3.7]},
        'no_training': True,
    }, indent=2)+'\n')
    print('Rendered H1/H2 in {} using {}'.format(OUT, FONT))


if __name__ == '__main__':
    main()
