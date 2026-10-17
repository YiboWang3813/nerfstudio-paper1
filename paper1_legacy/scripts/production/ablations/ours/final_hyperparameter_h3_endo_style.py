"""Reformat frozen CGAS sensitivity data to match the approved PLGR typography."""

import hashlib
import json
from decimal import Decimal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from final_hyperparameter_figures_ieee_tmi import (
    DATASETS, HEAT, ROOT, load, population_k, write_tsv)
from final_hyperparameter_h1_h2_endogaussian import FONT, style
from audit_panel_alignment import require_matplotlib_panel_alignment


OUT = ROOT / 'docs/production/final_hyperparameter_figures_20260916/ieee_tmi_endo_style'
SOURCE = ROOT / 'docs/production/final_paper_evidence_20260910/cga/hyperparameter_data.tsv'
SPINE = '#222222'
HEAT_NORM = TwoSlopeNorm(vmin=-.02, vcenter=0, vmax=.02)


def text_color(delta):
    red, green, blue = HEAT(HEAT_NORM(delta))[:3]
    luminance = .299 * red + .587 * green + .114 * blue
    return 'white' if luminance < .48 else 'black'


def frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE)
        spine.set_linewidth(.8)
    ax.tick_params(width=.65, length=2, pad=.6, labelsize=6.7,
                   color=SPINE, labelcolor=SPINE)


def heatmap(ax, rows, panel, y_values, x_values, default, ylabel, xlabel):
    selected = [r for r in rows if r['panel'] == panel]
    assert len(selected) == 18
    ys = [Decimal(v) for v in y_values]
    xs = [Decimal(v) for v in x_values]
    delta = np.empty((6, 3), dtype=float)
    population = np.empty((6, 3), dtype=object)
    plotted = []
    for ds_index, (dataset, name, *_) in enumerate(DATASETS):
        subset = [r for r in selected if r['dataset'] == dataset]
        assert len(subset) == 9
        baseline = next(r for r in subset if r['is_default'] == 'True')
        for row in subset:
            y, x = [Decimal(v) for v in row['point'].split('_')]
            assert y in ys and x in xs
            ri, ci = 3 * ds_index + ys.index(y), xs.index(x)
            difference = Decimal(row['final_psnr']) - Decimal(baseline['final_psnr'])
            assert abs(difference - Decimal(row['delta_psnr'])) < Decimal('1e-12')
            delta[ri, ci] = float(difference)
            population[ri, ci] = population_k(row['final_num_gaussians'])
            plotted.append(dict(
                dataset=name, panel=panel, y_value=str(y), x_value=str(x),
                row_id=row['row_id'],
                delta_psnr=str(difference if difference else Decimal(0)),
                final_G=row['final_num_gaussians'],
                annotation_G_k=population[ri, ci],
                is_default=str(row is baseline)))

    mesh = ax.pcolormesh(np.arange(4) - .5, np.arange(7) - .5, delta,
                         cmap=HEAT, norm=HEAT_NORM, shading='flat',
                         edgecolors='#F2F2F2', linewidth=.45,
                         rasterized=False)
    for ri in range(6):
        for ci in range(3):
            ax.text(ci, ri, population[ri, ci], ha='center', va='center',
                    fontsize=6.6, color=text_color(delta[ri, ci]))
    for ds_index in (0, 1):
        ri = 3 * ds_index + ys.index(Decimal(default[0]))
        ci = xs.index(Decimal(default[1]))
        ax.add_patch(Rectangle((ci - .47, ri - .47), .94, .94,
                               fill=False, edgecolor='black', linewidth=.7))
    ax.axhline(2.5, color='#B9B9B9', linewidth=.8)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(5.5, -.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(x_values)
    ax.set_yticks(range(6))
    # The numerical tick labels remain beside each panel. A single shared
    # L/L/L/A/A/A column is placed to the right of both heatmaps below.
    ax.set_yticklabels([y_values[i % 3] for i in range(6)])
    ax.set_xlabel(xlabel, labelpad=2.0, fontsize=7.8)
    ax.set_ylabel(ylabel, labelpad=.5, fontsize=7.8)
    if panel == 'qd_by_B':
        ax.yaxis.set_label_coords(-.25, .5)
    frame(ax)
    return mesh, plotted


def tau_strip(ax, rows):
    selected = [r for r in rows if r['panel'] == 'tau_alpha']
    assert len(selected) == 6
    values = [Decimal('.00005'), Decimal('.0001'), Decimal('.0005')]
    delta = np.empty((2, 3), dtype=float)
    population = np.empty((2, 3), dtype=object)
    plotted = []
    for ds_index, (dataset, name, *_) in enumerate(DATASETS):
        subset = [r for r in selected if r['dataset'] == dataset]
        baseline = next(r for r in subset if r['is_default'] == 'True')
        for row in subset:
            x = Decimal(row['point'])
            assert x in values
            ci = values.index(x)
            difference = Decimal(row['final_psnr']) - Decimal(baseline['final_psnr'])
            assert abs(difference - Decimal(row['delta_psnr'])) < Decimal('1e-12')
            delta[ds_index, ci] = float(difference)
            population[ds_index, ci] = population_k(row['final_num_gaussians'])
            plotted.append(dict(
                dataset=name, panel='tau_alpha', y_value='', x_value=str(x),
                row_id=row['row_id'],
                delta_psnr=str(difference if difference else Decimal(0)),
                final_G=row['final_num_gaussians'],
                annotation_G_k=population[ds_index, ci],
                is_default=str(row is baseline)))
    ax.pcolormesh(np.arange(4) - .5, np.arange(3) - .5, delta,
                  cmap=HEAT, norm=HEAT_NORM, shading='flat',
                  edgecolors='#F2F2F2', linewidth=.45, rasterized=False)
    for ri in range(2):
        for ci in range(3):
            ax.text(ci, ri, population[ri, ci], ha='center', va='center',
                    fontsize=6.6, color=text_color(delta[ri, ci]))
        ax.add_patch(Rectangle((.53, ri - .47), .94, .94,
                               fill=False, edgecolor='black', linewidth=.7))
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(1.5, -.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['5e-5', '1e-4', '5e-4'])
    ax.set_yticks([])
    ax.set_xlabel(r'Amplitude threshold $\tau_\alpha$', labelpad=1,
                  fontsize=7.8)
    frame(ax)
    return plotted


def add_shared_dataset_column(right_ax, strip_ax):
    # One bracket and key span each three-row dataset block on the right.
    for top, bottom, center, letter in [(-.45, 2.45, 1, 'L'),
                                        (2.55, 5.45, 4, 'A')]:
        right_ax.plot([2.55, 2.69, 2.69, 2.55],
                      [top, top, bottom, bottom], color=SPINE,
                      linewidth=.75, clip_on=False)
        right_ax.text(2.84, center, letter, ha='center', va='center',
                      fontsize=6.8, clip_on=False)
    for row, letter in enumerate('LA'):
        strip_ax.text(2.60, row, letter, ha='center', va='center',
                      fontsize=6.8, clip_on=False)


def render(rows, source_hash):
    assert {r['panel'] for r in rows} == {'qd_by_B', 'qp_by_qA', 'tau_alpha'}
    fig = plt.figure(figsize=(3.5, 2.0), facecolor='white')
    left = fig.add_axes([.12, .43, .275, .535])
    right = fig.add_axes([.52, .43, .275, .535])
    strip = fig.add_axes([.12, .145, .675, .145])
    cax = fig.add_axes([.88, .43, .020, .535])

    _, left_rows = heatmap(left, rows, 'qd_by_B',
                           ['0.96', '0.98', '0.99'], ['1e-3', '5e-3', '5e-2'],
                           ('.98', '.05'),
                           r'Densification quantile $q_d$',
                           r'Densification budget $B$')
    mesh, right_rows = heatmap(right, rows, 'qp_by_qA',
                               ['0.05', '0.10', '0.15'],
                               ['0.70', '0.80', '0.90'],
                               ('.10', '.80'),
                               r'Pruning quantile $q_p$',
                               r'Path protection $q_A$')
    bottom_rows = tau_strip(strip, rows)
    add_shared_dataset_column(right, strip)

    cb = fig.colorbar(mesh, cax=cax, ticks=[-.02, 0, .02])
    cb.solids.set_rasterized(False)
    cb.ax.set_yticklabels(['−0.02', '0', '0.02'])
    cb.ax.tick_params(labelsize=6.5, width=.5, length=1.8, pad=.3)
    cb.ax.yaxis.set_label_position('left')
    cb.set_label(r'$\Delta$PSNR (dB)', fontsize=7.0, labelpad=2.5)
    cb.outline.set_linewidth(.7)
    fig.text(.835, .29, 'L: Liver\nA: Arm\nText: #G (k)',
             ha='left', va='top', fontsize=6.7, linespacing=1.08)

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / 'H3_CGAS'
    require_matplotlib_panel_alignment(
        fig, axes=[left, right, strip], panel_ids=['a', 'b', 'c'],
        row_groups=[['a', 'b']], exclude_axes=[cax],
        json_out=str(base) + '.alignment.json',
        overlay_svg=str(base) + '.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.width and box.height and (
                box.x0 < -1 or box.y0 < -1 or
                box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1
            ):
                outside.append((artist.get_text(), box.bounds))
    assert not outside, outside
    fig.savefig(str(base) + '.pdf', facecolor='white',
                metadata={'Creator': 'ULGSplat frozen sensitivity data',
                          'CreationDate': None})
    fig.savefig(str(base) + '.png', dpi=600, facecolor='white',
                transparent=False)
    fig.savefig(str(base) + '.svg', facecolor='white')
    plt.close(fig)
    write_tsv(OUT / 'H3_CGAS_plotted_data.tsv',
              left_rows + right_rows + bottom_rows)
    (OUT / 'H3_CGAS_source_manifest.json').write_text(json.dumps({
        'source_data_path': str(SOURCE),
        'source_data_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'P2_source_sha256': source_hash,
        'font_requested': 'Times New Roman',
        'font_embedded': FONT,
        'size_inches': [3.5, 2.0],
        'color_scale_delta_psnr_dB': [-.02, .02],
        'no_training': True,
    }, indent=2) + '\n')
    print('Rendered {} with {}'.format(base, FONT))


if __name__ == '__main__':
    style()
    data, _, p2_hash = load()
    render(data['H3'], p2_hash)
