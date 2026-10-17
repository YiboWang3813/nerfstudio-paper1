"""Make one traceable Markdown inventory of the four frozen sensitivity figures."""

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / 'docs/production/final_paper_evidence_20260910'
FIGURES = ROOT / 'docs/production/final_hyperparameter_figures_20260916'
DEST = FIGURES / 'FOUR_FIGURES_RAW_DATA.md'
FILES = {
    'H1': EVIDENCE / 'ppp/rho_data.tsv',
    'H2': EVIDENCE / 'lgr/hyperparameter_data.tsv',
    'H3': EVIDENCE / 'cga/hyperparameter_data.tsv',
    'H4 structural': EVIDENCE / 'cga/structural_sources.tsv',
    'H4 plot': FIGURES / 'H4_CGAS_population/plotted_data.tsv',
    'H3 selection': FIGURES / 'H3_CGAS/selection_behavior_audit.tsv',
}
DATASET_NAMES = {'synthetic_liver': 'Liver', 'tus_rec_2024_qc': 'Arm'}
METRICS = ('psnr', 'l1', 'ssim', 'lpips', 'dists', 'gmsd')


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def link(path, label=None):
    target = path.relative_to(ROOT).as_posix()
    return '[{}](/{}{})'.format(label or path.name, ROOT.as_posix().strip('/'), '/' + target)


def table(headers, rows):
    result = ['| ' + ' | '.join(headers) + ' |',
              '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        assert len(row) == len(headers)
        result.append('| ' + ' | '.join(str(value).replace('|', '\\|') for value in row) + ' |')
    return '\n'.join(result)


def metric_values(row):
    return [row['final_' + name] for name in METRICS]


def population(row):
    return str(int(Decimal(row['final_num_gaussians'])))


def decimal_string(value):
    return '0' if value == 0 else str(value)


def config_link(row_id, run_index):
    entry = run_index[row_id]
    path = EVIDENCE / entry['packaged_config']
    assert path.is_file(), path
    return link(path, row_id)


def per_frame_link(row):
    path = Path(row['output_dir']) / 'eval/per_frame_metrics_iteration_030000.tsv'
    assert path.is_file(), path
    return link(path, 'frames')


def metric_section(rows, run_index, parameter, *, parameter_label, default,
                   figure_delta=True):
    headings = ['Dataset', parameter_label, 'Run/config', 'Per-frame',
                'PSNR', 'L1', 'SSIM', 'LPIPS', 'DISTS', 'GMSD', '#G',
                'ΔPSNR', 'ΔLPIPS']
    result = []
    for dataset in DATASET_NAMES:
        subset = [row for row in rows if row['dataset'] == dataset]
        if parameter == 'rho':
            subset.sort(key=lambda row: Decimal(row['rho']))
            origin = next(row for row in subset if Decimal(row['rho']) == Decimal('1'))
        else:
            subset.sort(key=lambda row: Decimal(row['point']))
            origin = next(row for row in subset if row['is_default'] == 'True')
        assert origin
        for row in subset:
            x = row[parameter]
            mark = ' **default**' if row is origin else ''
            delta_psnr = Decimal(row['final_psnr']) - Decimal(origin['final_psnr'])
            delta_lpips = Decimal(row['final_lpips']) - Decimal(origin['final_lpips'])
            result.append([DATASET_NAMES[dataset], x + mark,
                           config_link(row['row_id'], run_index),
                           per_frame_link(row)] + metric_values(row) +
                          [population(row), decimal_string(delta_psnr),
                           decimal_string(delta_lpips)])
            if figure_delta:
                assert abs(delta_psnr - Decimal(row['delta_psnr'])) < Decimal('1e-12')
                assert abs(delta_lpips - Decimal(row['delta_lpips'])) < Decimal('1e-12')
    return table(headings, result)


def grid_section(rows, run_index, panel, parameter_label):
    headings = ['Dataset', parameter_label, 'Run/config', 'Per-frame',
                'PSNR', 'L1', 'SSIM', 'LPIPS', 'DISTS', 'GMSD', '#G',
                'ΔPSNR', 'ΔLPIPS']
    output = []
    for dataset in DATASET_NAMES:
        subset = [row for row in rows if row['dataset'] == dataset and row['panel'] == panel]
        if panel == 'tau_alpha':
            subset.sort(key=lambda row: Decimal(row['point']))
        else:
            subset.sort(key=lambda row: tuple(Decimal(part) for part in row['point'].split('_')))
        origin = next(row for row in subset if row['is_default'] == 'True')
        for row in subset:
            point = row['point'].replace('_', ', ')
            if row is origin:
                point += ' **default**'
            delta_psnr = Decimal(row['final_psnr']) - Decimal(origin['final_psnr'])
            delta_lpips = Decimal(row['final_lpips']) - Decimal(origin['final_lpips'])
            assert abs(delta_psnr - Decimal(row['delta_psnr'])) < Decimal('1e-12')
            assert abs(delta_lpips - Decimal(row['delta_lpips'])) < Decimal('1e-12')
            output.append([DATASET_NAMES[dataset], point,
                           config_link(row['row_id'], run_index),
                           per_frame_link(row)] + metric_values(row) +
                          [population(row), decimal_string(delta_psnr),
                           decimal_string(delta_lpips)])
    return table(headings, output)


def main():
    run_index = {row['row_id']: row for row in read(EVIDENCE / 'audits/run_sources.tsv')}
    h1, h2, h3, h4 = (read(FILES[key]) for key in ('H1', 'H2', 'H3', 'H4 plot'))
    structural = read(FILES['H4 structural'])
    assert (len(h1), len(h2), len(h3), len(h4), len(structural)) == (14, 40, 42, 46, 10)
    assert {row['sweep'] for row in h1 + h2 + h3 if row['dataset'] == 'synthetic_liver'} == {'l1'}
    assert len({row['sweep'] for row in h1 + h2 + h3 if row['dataset'] == 'tus_rec_2024_qc'}) == 1
    assert len({(r['dataset'], r['output_dir']) for r in h4}) == 46
    assert {(r['dataset'], r['output_dir']) for r in structural}.issubset(
        {(r['dataset'], r['output_dir']) for r in h4})
    manifest = json.loads((EVIDENCE / 'audits/final_data_source_manifest.json').read_text())
    source_audit = json.loads((FIGURES / 'source_audit.json').read_text())
    assert manifest['P2_source_sha256'] == source_audit['P2_source_sha256']
    for key in ('H1', 'H2', 'H3', 'H4 structural'):
        path = FILES[key]
        rel = path.relative_to(EVIDENCE).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest['package_file_sha256'][rel] == digest
    for key, rows in (('H1', h1), ('H2', h2), ('H3', h3)):
        for row in rows:
            entry = run_index[row['row_id']]
            assert entry['output_dir'] == row['output_dir']
            assert entry['validated_p2_source_sha256'] == manifest['P2_source_sha256']
            assert entry['execution_source_sha256'] == row['execution_source_sha256']

    lines = [
        '# H1–H4 超参数图：完整原始数值与来源', '',
        '本文件汇集四张图使用的**逐 run 最终指标**，不做论文显示精度的四舍五入。'
        '图中的曲线/热图使用这些指标相对于各自默认点的差值。'
        '逐帧记录位于每条 run 的 `eval/per_frame_metrics_iteration_030000.tsv`；'
        '每行的 **frames** 链接直接指向它。', '',
        '数据来自最终 Fixed finite-slice + Full global 的冻结实验。H1 使用 AFGM；'
        'H2/H3 固定 MP，以隔离渲染与适应参数。H4 合并 CGAS 参数扫描和结构消融；'
        '其中结构对照本来就采用不同 CGAS 策略。数据集是 representative Liver `l1` 与 '
        'Arm `010_LH_Per_L_DtP_qc_0005_0370`，并非主比较的 78-case 聚合。'
        '所有数据经过 P2 配置等价性核查。', '',
        '## 数据位置与谱系', '',
        table(['图', '权威逐 run 汇总', '作图输入', '记录数'], [
            ['H1 AFGM', link(FILES['H1']), link(FIGURES / 'H1_AFGM/plotted_data.tsv'), '14（Liver/Arm 各 7）'],
            ['H2 PLGR', link(FILES['H2']), link(FIGURES / 'H2_PLGR/plotted_data.tsv'), '40（5 参数，共 20 点/数据集）'],
            ['H3 CGAS', link(FILES['H3']), link(FIGURES / 'H3_CGAS/plotted_data.tsv'), '42（2 个 3×3 网格 + 3 点扫描，每数据集 21 个配置成员）'],
            ['H4 CGAS population', link(FILES['H3']) + ' + ' + link(FILES['H4 structural']),
             link(FILES['H4 plot']), '46 个不同的物理 run（每数据集 23）'],
        ]), '',
        '每个 run 的 `row_id`、执行源码 SHA、等价 P2 SHA、输出目录与归档配置，见 '
        + link(EVIDENCE / 'audits/run_sources.tsv') + '。源文件 SHA 清单见 '
        + link(EVIDENCE / 'audits/final_data_source_manifest.json') + '；作图时的配置/数值审计见 '
        + link(FIGURES / 'source_audit.json') + '。P2 源码归档 SHA-256：'
        + '`' + manifest['P2_source_sha256'] + '`。', '',
        '**读表说明：**PSNR 单位 dB，#G 为最终 Gaussian 个数；'
        'L1/LPIPS/DISTS/GMSD 越低越好，PSNR/SSIM 越高越好。'
        'Δ 为该数据集同一参数扫描内 `当前值 − 默认值`，'
        '根据保留的原始十进制字段重算。默认点可由同一个物理 run 在多个扫描中复用；'
        '它不是额外一次训练。每个参数点是一条训练 run，不能把点间波动当作多 seed 方差。', '',
        '## H1 — AFGM spacing scale ρ', '',
        '默认 ρ=1；采样 0.60、0.75、0.90、1.00、1.10、1.25、1.50。'
        '主图画 ΔPSNR；LPIPS 原值和 ΔLPIPS 在下表，供检查指标权衡。', '',
        metric_section(h1, run_index, 'rho', parameter_label='ρ', default='1'), '',
        '## H2 — PLGR 五项参数', '',
        '全部为 Fixed local + Full ray integral；sigma_ele=1 mm 是最终固定 slice width。'
        '主图画各参数的 ΔPSNR，LPIPS review 图使用相同采样点。', '',
    ]
    panels = [
        ('sigma_ele', 'finite-slice width σ_ele (mm)', '0.8, 1.0, 1.5, 2.0', '1.0'),
        ('beta', 'composition weight β', '0.4, 0.6, 0.8, 1.0', '0.8'),
        ('lambda_ray', 'attenuation λ_ray', '0.20, 0.35, 0.55, 0.75, 1.00', '0.55'),
        ('Nq', 'quadrature order Nq', '1, 2, 3', '3'),
        ('half_window', 'integration half-window w (mm)', '1, 2, 3, 4', '2'),
    ]
    for index, (panel, title, values, default) in enumerate(panels, 1):
        subset = [row for row in h2 if row['panel'] == panel]
        assert len(subset) in (6, 8, 10)
        lines += ['### H2.{} {}\n'.format(index, title),
                  '采样：{}；默认：{}。'.format(values, default), '',
                  metric_section(subset, run_index, 'point', parameter_label=title, default=default), '']

    lines += [
        '## H3 — CGAS 参数', '',
        '热图单元格画 ΔPSNR；矩阵下方保留所有未经四舍五入的质量指标和 #G。'
        '这里的 B 实际扫描值是 0.001/0.005/0.05，不能用早期草案的 0.02/0.05/0.08 替换。', '',
    ]
    for panel, title, sampled, default in [
        ('qd_by_B', 'q_d × B', 'q_d={0.96, 0.98, 0.99}; B={0.001, 0.005, 0.05}', '(0.98, 0.05)'),
        ('qp_by_qA', 'q_p × q_A', 'q_p={0.05, 0.10, 0.15}; q_A={0.70, 0.80, 0.90}', '(0.10, 0.80)'),
        ('tau_alpha', 'opacity pruning threshold τ_α', '{5e-5, 1e-4, 5e-4}', '1e-4'),
    ]:
        lines += ['### ' + title, '', '采样：{}；默认：{}。'.format(sampled, default), '',
                  grid_section(h3, run_index, panel, title), '']
    lines += [
        '选择行为额外证据：' + link(FILES['H3 selection']) + '。'
        '`q_p × q_A` 共 18 个配置、216 次 adaptation event，其局部贡献阈值全部为 0；'
        '因此该范围中的 q_p 处于零值并列造成的饱和区。q_A 确实改变 path protection；'
        'B 在低预算时可能成为候选限制因素。', '',
        '## H4 — CGAS quality–population scatter', '',
        'H4 无新训练：从 H3 的参数扫描和 CGAS 的 5 行结构消融中按'
        '`(dataset, output_dir)` 去重。下表 46 行，每数据集 23 个不同物理 run。'
        '“成员”列保留一个 run 被多个 panel 当作默认点复用的信息。'
        '结构行包含 GD/RGD 等策略对照，不能全部称作 Full CGAS。', '',
    ]
    h4_sources = {(row['dataset'], row['output_dir']): row for row in h3 + structural}
    scatter_rows = []
    for dataset in DATASET_NAMES:
        for row in sorted((r for r in h4 if r['dataset'] == dataset),
                          key=lambda r: (Decimal(r['G_k']), r['row_id'])):
            matching = run_index[row['row_id']]
            assert matching['output_dir'] == row['output_dir']
            source = h4_sources[(row['dataset'], row['output_dir'])]
            assert row['PSNR'] == source['final_psnr']
            assert row['LPIPS'] == source['final_lpips']
            scatter_rows.append([DATASET_NAMES[dataset],
                                 config_link(row['row_id'], run_index),
                                 per_frame_link(row)] + metric_values(source) +
                                [str(int(Decimal(row['final_G']))), row['G_k'],
                                 row['structural_variant'] or 'sensitivity',
                                 row['is_default'], row['memberships']])
    lines += [table(['Dataset', 'Run/config', 'Per-frame', 'PSNR', 'L1', 'SSIM',
                     'LPIPS', 'DISTS', 'GMSD', '#G', '#G (k)', '点类型', '默认', '成员'],
                    scatter_rows), '',
              'H4 使用各数据集自身 PSNR 纵轴尺度，未拟合趋势线。'
              'RGD 无 pruning 的约 1.05M/1.02M Gaussians 延伸了参数扫描的数量范围；'
              '散点不能单独证明“越少越好”或因果单调关系。', '',
              '## 数据完整性检查', '',
              '- H1/H2/H3 权威表分别为 14/40/42 行；H4 有 46 个去重的物理 run。',
              '- 每条逐帧文件、归档 Hydra 配置均存在；run ID、输出目录和源码 SHA 与总索引对应。',
              '- H1/H2/H3 的 ΔPSNR/ΔLPIPS 与封存差值逐项一致，误差低于 1e-12。',
              '- 权威 TSV 的 SHA-256 与封存 manifest 一致。',
              '- 图形目录中的 `plotted_data.tsv` 是直接作图数据；以上权威 TSV 才是逐 run 汇总来源。',
              '- H4 的 `plotted_data.tsv` 已完成跨面板去重，原始成员仍能经 run ID 回溯。', '']
    DEST.write_text('\n'.join(lines))
    print('{}: {} lines, {} bytes'.format(DEST, len(lines), DEST.stat().st_size))


if __name__ == '__main__':
    main()
