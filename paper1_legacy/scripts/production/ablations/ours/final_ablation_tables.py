"""Read-only frozen evidence audit; writes only a new textual evidence package."""
import csv
import hashlib
import itertools
import json
import re
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'docs/production/final_paper_evidence_20260910'
OUT = ROOT / 'docs/production/final_ablation_tables_20260916'
DS = {'synthetic_liver': 'Liver', 'tus_rec_2024_qc': 'Arm', 'tdsc_abus_2023': 'Breast'}
MET = ('PSNR', 'SSIM', 'LPIPS')
getcontext().prec = 40
D = Decimal
INPUTS = {}
CONFIGS = {}
CHECKS = []
RANKS = []
TIES = []


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(rel):
    p = SOURCE / rel
    INPUTS[rel] = sha(p)
    if p.suffix == '.json':
        return json.loads(p.read_text())
    with p.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write(name, rows):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w = csv.DictWriter(f, keys, delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def doc(name, text):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + '\n')


def md(rows, keys=None):
    keys = keys or list(rows[0])
    def fmt(v):
        return f'{v:.9f}'.rstrip('0').rstrip('.') if isinstance(v, D) else str(v)
    return '| ' + ' | '.join(keys) + ' |\n| ' + ' | '.join(['---'] * len(keys)) + ' |\n' + '\n'.join('| ' + ' | '.join(fmt(r[k]) for k in keys) + ' |' for r in rows) + '\n'


def mean(values):
    values = list(values)
    return sum(values, D(0)) / len(values)


def sd(values):
    values = list(values)
    mu = mean(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)).sqrt()


def verify(value, frozen, context):
    error = abs(D(value) - D(frozen))
    assert error < D('5e-13'), (context, value, frozen)
    CHECKS.append(dict(context=context, recomputed=value, frozen=frozen, absolute_difference=error))


def load_sources(rel, section):
    rows = read(rel)
    cleaned = []
    for r in rows:
        cfgrel = 'audits/resolved_configs/structural_main_' + r['row_id'] + '.yaml'
        if r['row_id'] not in CONFIGS:
            p = SOURCE / cfgrel
            INPUTS[cfgrel] = sha(p)
            CONFIGS[r['row_id']] = yaml.safe_load(p.read_text())
        c = CONFIGS[r['row_id']]
        sr = c['model']['slice_render']
        dc = c['model']['offline']['density_control']
        loading = c['scene']['loading']
        glob = sr['ray_attenuation_formula'] if sr['ray_attenuation_enabled'] else 'none'
        prune = dc['pruning_policy']
        effective = 'local_aware' if prune == 'path_aware' and glob == 'none' else prune
        assert c['system']['seed'] == 42 and c['engine']['iterations'] == 30000
        assert c['scene']['sampler']['num_init'] == 1000000
        dataset_key = 'tus_rec_2024' if r['dataset'] == 'tus_rec_2024_qc' else r['dataset']
        assert c['scene']['dataset_name'] == dataset_key and loading['sweep_name'] == r['sweep']
        if r['dataset'] == 'tus_rec_2024_qc':
            assert loading['dataset_dirname'] == 'tus_rec_2024_qc'
        label = r['label']
        a = None
        if section == 'joint':
            l, g, a = int(label[1]), int(label[3]), int(label[5])
            local, global_mode = ('fixed', 'integral') if l else ('none', 'none')
            dense, pruning = ('residual_guided', 'path_aware' if l else 'local_aware') if g else ('gradient', 'opacity')
            pose, spacing = ('canonical_sweep', 'uniform') if a else ('file', 'uniform')
            newlabel = f'A{a}P{l}C{g}'
        elif section == 'plgr':
            local, global_mode = {'Plane': ('none', 'none'), 'Fixed': ('fixed', 'none'), 'Adaptive': ('adaptive', 'none'), 'Fixed+Simple': ('fixed', 'slice_depth'), 'Fixed+Full': ('fixed', 'integral')}[label]
            dense, pruning = 'residual_guided', 'local_aware' if global_mode == 'none' else 'path_aware'
            pose, spacing = 'file', 'uniform'
            newlabel = label.replace('+', ' + ')
        elif section == 'cgas':
            local, global_mode = 'fixed', 'integral'
            dense, pruning = {'GD': ('gradient', 'opacity'), 'RGD': ('residual_guided', 'none'), 'RGD+OP': ('residual_guided', 'opacity'), 'RGD+LP': ('residual_guided', 'local_aware'), 'Full': ('residual_guided', 'path_aware')}[label]
            pose, spacing = 'file', 'uniform'
            newlabel = 'Full CGAS' if label == 'Full' else label.replace('+', ' + ')
        else:
            local, global_mode, dense, pruning = 'fixed', 'integral', 'residual_guided', 'path_aware'
            pose = 'file' if label == 'MP' else 'canonical_sweep'
            spacing = {'MP': 'uniform', 'PPP': 'uniform', 'PP-S': 'file_step_norm', 'PP-A': 'file_axis_projection'}[label]
            newlabel = 'AFGM' if label == 'PPP' else label
        assert (sr['local_conditioning_mode'], glob, dc['densification_policy'], effective, loading['pose_mode'], loading['canonical_sweep_spacing_mode']) == (local, global_mode, dense, pruning, pose, spacing), r['row_id']
        assert sr['us_focus_sigma_ele_0'] == 1.0
        assert loading['canonical_sweep_delta_scale'] == 1.0
        if global_mode != 'none':
            assert sr['ray_attenuation_source'] == 'attenuation'
            assert sr['ray_attenuation_beta'] == .8 and sr['ray_attenuation_strength'] == .55
            assert sr['ray_attenuation_quadrature_points'] == 3 and sr['ray_attenuation_quadrature_window_mm'] == 2.
        if pruning == 'path_aware':
            assert dc['responsibility']['protect_attenuation_mode'] == 'path_contribution'
        item = dict(dataset=DS[r['dataset']], case_id=r['sweep'], variant=newlabel)
        if section == 'joint':
            item.update(AFGM=a, PLGR=l, CGAS=g)
        item.update({m: D(r['final_' + m.lower()]) for m in MET})
        item.update(final_G=D(r['final_num_gaussians']), G_k=D(r['final_num_gaussians']) / 1000,
                    row_id=r['row_id'], output_dir=r['output_dir'], packaged_config=str(SOURCE / cfgrel),
                    source_file=str(SOURCE / rel), execution_source_sha256=r['execution_source_sha256'],
                    equivalent_final_source_sha256=r['validated_equivalent_p2_sha256'],
                    local_mode=local, global_mode=glob, densification_policy=dense,
                    configured_pruning_policy=prune, effective_pruning_policy=effective,
                    pose_mode=pose, spacing_mode=spacing)
        cleaned.append(item)
    return cleaned


def wide(rows, labels, metrics=MET):
    result = []
    for label in labels:
        item = dict(variant=label)
        for ds in ('Liver', 'Arm'):
            sub = [r for r in rows if r['dataset'] == ds and r['variant'] == label]
            assert sub
            for m in metrics:
                item[ds + '_' + m] = mean(r[m] for r in sub)
        result.append(item)
    return result


def compare_wide(rows, rel, aliases=None):
    aliases = aliases or {}
    old = read(rel)
    for r in rows:
        o = next(x for x in old if x['variant'] == aliases.get(r['variant'], r['variant']))
        for k, value in r.items():
            if k not in ('variant', 'AFGM', 'PLGR', 'CGAS'):
                verify(value, o[k], rel + ':' + r['variant'] + ':' + k)


def display(section, rows, leading=('variant',)):
    """Use manuscript 3-decimal metrics and integer thousands; rank raw quality only."""
    metrics = [k for k in rows[0] if k not in leading and k != 'variant']
    rendered, tex = [], []
    ranks = {}
    for k in metrics:
        if k.endswith('G_k'):
            continue
        vals = sorted(set(r[k] for r in rows), reverse=not k.endswith('LPIPS'))
        buckets = defaultdict(list)
        for r in rows:
            rank = 1 + sum(v > r[k] if not k.endswith('LPIPS') else v < r[k] for v in (x[k] for x in rows))
            ranks[r['variant'], k] = rank
            RANKS.append(dict(table=section, variant=r['variant'], metric=k, raw=r[k], display=f'{r[k]:.3f}', rank=rank, mark='bold' if rank == 1 else 'underline' if r[k] == vals[1] else 'plain'))
            buckets[f'{r[k]:.3f}'].append(r)
        for v, group in buckets.items():
            if len(group) > 1:
                TIES.append(dict(table=section, metric=k, displayed=v, variants='; '.join(r['variant'] for r in group), raw_values='; '.join(str(r[k]) for r in group), exact_tie=len(set(r[k] for r in group)) == 1))
    for r in rows:
        item = {k: r[k] for k in leading}
        cells = [str(r[k]) for k in leading]
        for k in metrics:
            value = f'{r[k]:.0f}' if k.endswith('G_k') else f'{r[k]:.3f}'
            rank = ranks.get((r['variant'], k), 0)
            item[k] = '**' + value + '**' if rank == 1 else '<u>' + value + '</u>' if rank == 2 else value
            cells.append('\\textbf{' + value + '}' if rank == 1 else '\\underline{' + value + '}' if rank == 2 else value)
        rendered.append(item)
        tex.append(' & '.join(cells) + r' \\')
    headers = list(rendered[0])
    nice = lambda k: k.replace('_G_k', ' #G (k)').replace('_PSNR', ' PSNR ↑').replace('_SSIM', ' SSIM ↑').replace('_LPIPS', ' LPIPS ↓').replace('variant', 'Variant')
    rendered = [{nice(k): r[k] for k in headers} for r in rendered]
    note = 'Quality: bold = raw-value best; underline = raw-value second-best. Display precision follows the existing manuscript: 3 decimals for quality; integer thousands for population. Ranking is descriptive, not significance. Population is unranked.'
    doc(section + '/table.md', '# ' + section.upper() + '\n\n' + note + '\n\n' + md(rendered))
    doc(section + '/table_latex.tex', '% Evidence fragment only. Columns: ' + ', '.join(leading + tuple(metrics)) + '\n% ' + note + '\n' + '\n'.join(tex))
    return rendered


def transitions(rows, pairs, metrics=MET):
    out = []
    for ds in sorted(set(r['dataset'] for r in rows)):
        for before, after in pairs:
            aa = {r['case_id']: r for r in rows if r['dataset'] == ds and r['variant'] == before}
            bb = {r['case_id']: r for r in rows if r['dataset'] == ds and r['variant'] == after}
            assert aa.keys() == bb.keys()
            for m in metrics:
                delta = [bb[c][m] - aa[c][m] for c in aa]
                tol = D('0.000001')
                win = sum(v < -tol if m == 'LPIPS' else v > tol for v in delta)
                tie = sum(abs(v) <= tol for v in delta)
                out.append(dict(dataset=ds, transition=before + ' -> ' + after, metric=m, n=len(delta), mean_delta=mean(delta), std_delta=sd(delta) if len(delta) > 1 else '', wins=win, ties=tie, losses=len(delta) - win - tie))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read('audits/final_data_source_manifest.json')
    joint = load_sources('joint/final_joint_sources.tsv', 'joint')
    afgm = load_sources('ppp/final_ppp_sources.tsv', 'afgm')
    multi = load_sources('ppp/multisweep_sources.tsv', 'afgm')
    plgr = load_sources('lgr/structural_sources.tsv', 'plgr')
    cgas = load_sources('cga/structural_sources.tsv', 'cgas')
    assert list(map(len, (joint, afgm, multi, plgr, cgas))) == [16, 8, 32, 55, 10]
    # All table metrics come from the frozen per-run source rows, never old TeX.
    for name, data in [('joint/raw.tsv', joint), ('afgm/raw_representative.tsv', afgm), ('afgm/raw_multisweep.tsv', multi), ('plgr/raw.tsv', plgr), ('cgas/raw.tsv', cgas)]:
        write(name, data)
    for ds, n in [('Liver', 6), ('Arm', 5), ('Breast', 5)]:
        sets = [{r['case_id'] for r in multi if r['dataset'] == ds and r['variant'] == v} for v in ['MP', 'AFGM']]
        assert len(sets[0]) == n and sets[0] == sets[1]
        if ds != 'Breast':
            assert all({r['case_id'] for r in plgr if r['dataset'] == ds and r['variant'] == v} == sets[0] for v in ('Plane', 'Fixed', 'Adaptive', 'Fixed + Simple', 'Fixed + Full'))
    for ds in ['Liver', 'Arm']:
        rep = {r['case_id'] for r in joint if r['dataset'] == ds}
        assert len(rep) == 1
        assert all({r['case_id'] for r in data if r['dataset'] == ds} == rep for data in [afgm, cgas])
    jlabels = [f'A{a}P{p}C{c}' for p, c, a in itertools.product((0, 1), repeat=3)]
    jw = wide(joint, jlabels)
    for r in jw:
        r.update(AFGM=int(r['variant'][1]), PLGR=int(r['variant'][3]), CGAS=int(r['variant'][5]))
    compare_wide(jw, 'joint/final_joint_table.tsv', {v: f'L{v[3]}C{v[5]}P{v[1]}' for v in jlabels})
    aw = wide(afgm, ['MP', 'PP-S', 'PP-A', 'AFGM'])
    compare_wide(aw, 'ppp/final_ppp_table.tsv', {'AFGM': 'PPP'})
    pw = wide(plgr, ['Plane', 'Fixed', 'Adaptive', 'Fixed + Simple', 'Fixed + Full'])
    compare_wide(pw, 'lgr/final_lgr_table.tsv', {v: v.replace(' + ', '+') for v in ['Fixed + Simple', 'Fixed + Full']})
    cw = wide(cgas, ['GD', 'RGD', 'RGD + OP', 'RGD + LP', 'Full CGAS'], ('PSNR', 'LPIPS', 'G_k'))
    compare_wide(cw, 'cga/final_cga_table.tsv', {'Full CGAS': 'Full', 'RGD + OP': 'RGD+OP', 'RGD + LP': 'RGD+LP'})
    for section, rows in [('joint', jw), ('afgm', aw), ('plgr', pw), ('cgas', cw)]:
        write(section + '/aggregate_raw.tsv', rows)
        display(section, rows, ('AFGM', 'PLGR', 'CGAS') if section == 'joint' else ('variant',))
    ms = []
    oldmulti = read('ppp/final_ppp_multisweep.tsv')
    for ds, key in [('Liver', 'synthetic_liver'), ('Arm', 'tus_rec_2024_qc'), ('Breast', 'tdsc_abus_2023')]:
        for v in ('MP', 'AFGM'):
            rr = [r for r in multi if r['dataset'] == ds and r['variant'] == v]
            old = next(r for r in oldmulti if r['dataset'] == key and r['label'] == ('PPP' if v == 'AFGM' else v))
            item = dict(dataset=ds, variant=v, n=len(rr))
            for m in ('PSNR', 'SSIM'):
                for suffix, fun in [('mean', mean), ('std', sd)]:
                    k = m + '_' + suffix
                    item[k] = fun(r[m] for r in rr)
                    verify(item[k], old[k], 'AFGM multi ' + ds + v + k)
            ms.append(item)
    write('afgm/multisweep_summary_raw.tsv', ms)
    msdisp = [dict(dataset=r['dataset'], variant=r['variant'], n=r['n'], PSNR=f"{r['PSNR_mean']:.3f} ± {r['PSNR_std']:.3f}", SSIM=f"{r['SSIM_mean']:.3f} ± {r['SSIM_std']:.3f}") for r in ms]
    path = OUT / 'afgm/table.md'
    doc('afgm/table.md', path.read_text().replace('# AFGM', '# AFGM\n\n## (a) Representative diagnostic comparison') + '\n## (b) Multi-sweep MP versus AFGM\n\nEqual-sweep means ± sample standard deviation (ddof=1), not uncertainty across seeds. No best/second marking: this panel assesses preservation, not universal superiority.\n\n' + md(msdisp))
    texmulti = [r'\multicolumn{7}{l}{(b) Multi-sweep MP versus AFGM} \\', r'% Dataset & Variant & n & PSNR mean+-std (2 cols) & SSIM mean+-std (2 cols)',r'Dataset & Geometry & $n$ & \multicolumn{2}{c}{PSNR$\uparrow$} & \multicolumn{2}{c}{SSIM$\uparrow$} \\']
    for r in ms:
        texmulti.append(f"{r['dataset']} & {r['variant']} & {r['n']} & \\multicolumn{{2}}{{c}}{{${r['PSNR_mean']:.3f} \\pm {r['PSNR_std']:.3f}$}} & \\multicolumn{{2}}{{c}}{{${r['SSIM_mean']:.3f} \\pm {r['SSIM_std']:.3f}$}} " + r'\\')
    doc('afgm/table_latex.tex', '% Seven columns; two panels in one table.\n' + r'\multicolumn{7}{l}{(a) Representative diagnostic comparison} \\' + '\n' + (OUT / 'afgm/table_latex.tex').read_text() + '\n\\hline\n' + '\n'.join(texmulti))
    # Rounding collisions in the preservation panel are explicit, but not ranked.
    for ds in ['Liver', 'Arm', 'Breast']:
        rr = [r for r in ms if r['dataset'] == ds]
        for m in ['PSNR', 'SSIM']:
            if f"{rr[0][m+'_mean']:.3f}" == f"{rr[1][m+'_mean']:.3f}":
                TIES.append(dict(table='afgm_multisweep', metric=ds+'_'+m, displayed=f"{rr[0][m+'_mean']:.3f}", variants='MP; AFGM', raw_values='; '.join(str(r[m+'_mean']) for r in rr), exact_tie=rr[0][m+'_mean']==rr[1][m+'_mean']))
    effects = []
    for ds in ['Liver', 'Arm']:
        data = {r['variant']: r for r in joint if r['dataset'] == ds}
        for module, pos in [('AFGM', 1), ('PLGR', 3), ('CGAS', 5)]:
            for v, r in data.items():
                if v[pos] == '1':
                    continue
                on = v[:pos] + '1' + v[pos+1:]
                effects.append(dict(dataset=ds, module=module, off=v, on=on, **{'delta_'+m: data[on][m]-r[m] for m in MET}, delta_G=data[on]['final_G']-r['final_G']))
    write('joint/paired_component_effects.tsv', effects)
    es = [dict(dataset=ds, module=module, **{'mean_delta_'+m: mean(r['delta_'+m] for r in effects if r['dataset']==ds and r['module']==module) for m in MET}) for ds in ['Liver','Arm'] for module in ['AFGM','PLGR','CGAS']]
    write('joint/mean_component_effects.tsv', es)
    at = transitions(multi, [('MP','AFGM')], ('PSNR','SSIM'))
    pt = transitions(plgr, [('Plane','Fixed'),('Fixed','Adaptive'),('Fixed','Fixed + Simple'),('Fixed + Simple','Fixed + Full')])
    write('afgm/paired_multisweep.tsv', at)
    write('plgr/paired_transitions.tsv', pt)
    ct = []
    for ds in ['Liver','Arm']:
        data = {r['variant']:r for r in cgas if r['dataset']==ds}
        for a,z in [('GD','RGD'),('RGD','RGD + OP'),('RGD','RGD + LP'),('RGD + OP','RGD + LP'),('RGD + LP','Full CGAS')]:
            x,y=data[a],data[z]
            ct.append(dict(dataset=ds,transition=a+' -> '+z,delta_G=y['final_G']-x['final_G'],population_change_percent=(y['final_G']-x['final_G'])/x['final_G']*100,delta_PSNR=y['PSNR']-x['PSNR'],delta_LPIPS=y['LPIPS']-x['LPIPS']))
    write('cgas/transitions.tsv', ct)
    events=read('cga/event_totals.tsv')
    for r in events:
        r['dataset']=DS[r['dataset']]
        r['label']='Full CGAS' if r['label']=='Full' else r['label'].replace('+',' + ')
        assert D(r['final_G'])==1000000+D(r['total_spawned'])-D(r['total_pruned'])
    write('cgas/event_totals.tsv',events)
    # Original raw sources were validated against frame logs when frozen; no model evaluation here.
    current_config_audit(manifest)
    for rel,h in INPUTS.items():
        if rel!='audits/final_data_source_manifest.json':
            assert manifest['package_file_sha256'][rel]==h,rel
    assert all(r['equivalent_final_source_sha256']==manifest['P2_source_sha256'] for data in [joint,afgm,multi,plgr,cgas] for r in data)
    write('rank_display_audit.tsv',RANKS)
    write('rounding_ties.tsv',TIES)
    write('frozen_aggregate_verification.tsv',CHECKS)
    analyses(joint,afgm,multi,plgr,cgas,es,at,pt,ct,manifest)
    doc('source_manifest.json',json.dumps(dict(frozen_package=str(SOURCE),P2_source_sha256=manifest['P2_source_sha256'],input_sha256=INPUTS,unique_run_configs=len(CONFIGS),case_rows=dict(joint=16,afgm_representative=8,afgm_multisweep=32,plgr=55,cgas=10),aggregate_checks=len(CHECKS),max_aggregate_error=str(max(r['absolute_difference'] for r in CHECKS)),ranking_cells=len(RANKS),rounding_collision_groups=len(TIES),no_training=True,no_jobs=True,no_manuscript_edits=True),indent=2))
    text = '\n\n'.join((OUT / section / 'table.md').read_text() for section in ['joint','afgm','plgr','cgas'])
    doc('ALL_FOUR_TABLES.md',text)
    for section in ['joint','afgm','plgr','cgas']:
        for file in ['table.md','table_latex.tex']:
            assert not re.search(r'\b(?:PPP|LGR|CGA)\b',(OUT/section/file).read_text()),(section,file)
    print('PASS:',OUT,'case rows 16/8/32/55/10; aggregate checks',len(CHECKS),'unique configs',len(CONFIGS))


def current_config_audit(manifest):
    from omegaconf import OmegaConf
    from configs.setup import load_hydra_config
    matrix=read('audits/final_refresh_matrix.json')
    out=[]
    for rid in ['p2_009','p2_048']:
        row=next(r for r in matrix if r['batch1_id']==rid)
        current=OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',row['overrides']),resolve=True)
        old=CONFIGS[rid]
        fields=['model','scene','loss','engine','metric']
        assert all(current[k]==old[k] for k in fields) and current['system']['seed']==old['system']['seed']
        out.append(dict(row_id=rid,fields=fields+['system.seed'],status='EXACT MATCH',production_config='configs_hydra/experiment/production/ablations/ours/paper_semantic_freeze.yaml'))
    doc('current_config_audit.json',json.dumps(out,indent=2))


def analyses(joint,afgm,multi,plgr,cgas,effects,at,pt,ct,manifest):
    provenance='All paths below are relative to `docs/production/final_paper_evidence_20260910/`. New raw files carry exact absolute source/config/output paths, row IDs and execution/equivalence hashes. Values were recomputed from frozen per-run final metrics, not copied from old manuscript values. This is arithmetic reaggregation, not a new evaluation. No per-frame rerendering or metric recomputation was performed.'
    common='Representative protocol: Liver `l1`; Arm `010_LH_Per_L_DtP_qc_0005_0370`; 1M initial Gaussians, 30k iterations, seed 42, unchanged held-out split. Single-seed descriptive comparisons, not statistical significance. Arm sweeps/segments must not be treated as independent subjects.'
    doc('joint/analysis.md','# Joint component audit\n\n'+provenance+'\n\nAuthoritative raw: `joint/final_joint_sources.tsv`; frozen cross-check: `joint/final_joint_table.tsv`.\n\n'+common+'''

## Design meaning

- AFGM off = MP, full measured 6-DoF geometry; on = frame-order/nominal-spacing construction.
- PLGR off = Plane local/no global, NOT Fixed local-only. On = Fixed 1-mm local + Full causal integral global.
- CGAS off = gradient densification + opacity pruning. On = residual-guided/path-aware adaptation; without global statistics, pruning is local-aware. Some reused off rows explicitly encode local-aware; others request path-aware and use the implemented fallback. Their effective semantics agree.
- The three switches form the full eight backgrounds. Table columns are AFGM/PLGR/CGAS; old L/C/P codes were translated, not blindly reordered.

## Mean on-minus-off effects across four paired backgrounds

'''+md(effects)+'''
PLGR is the dominant PSNR contributor. These are four factorial backgrounds, not four independent replicates. CGAS chiefly concerns capacity allocation; the quality table alone does not prove spatial allocation, so consult its internal population/event evidence. AFGM removes measured geometry from trajectory construction, rather than acting as a universal accuracy booster: Liver LPIPS becomes worse when enabled, including the complete method. The full framework supports complementary roles, not a claim of positive synergy for every metric.

No joint qualitative figure is included or regenerated. Existing historical figures were not deleted or modified.
''')
    doc('afgm/analysis.md','# AFGM internal audit\n\n'+provenance+'\n\nSources: `ppp/final_ppp_sources.tsv`, `ppp/multisweep_sources.tsv`; cross-checks: `ppp/final_ppp_table.tsv`, `ppp/final_ppp_multisweep.tsv`. These are legacy path names only.\n\n'+common+'''

## Scientifically distinct rows

- MP: measured translations and rotations, complete 6-DoF geometry.
- PP-S: partial-oracle diagnostic. Adjacent measured translation Euclidean magnitudes accumulated along the predefined signed scan axis; rotations not used.
- PP-A: partial-oracle diagnostic. Signed adjacent measured translation projection on the predefined axis; no automatic axis estimation; rotations not used.
- AFGM: retained frame index, dataset-specific nominal spacing times rho=1, predefined signed axis and reference frame; identity rotations. Measured probe geometry is not used to construct its trajectory. A loader may still read a pose file for frame bookkeeping.

PP-S and PP-A are diagnostics, not submodules of AFGM. All four rows use final Fixed+Full PLGR and Full CGAS. Three metrics are retained in the diagnostic panel to expose perceptual trade-offs.

## Multi-sweep result

Liver 6, Arm 5, Breast 5: paired identical case IDs, equal-sweep arithmetic mean; sample std (ddof=1) across sweeps. The std is not a seed confidence interval. PSNR/SSIM only in this panel; no winner highlighting because its question is accuracy preservation.

'''+md(at)+'''
SUPPORTED within this protocol: AFGM removes measured-pose geometry dependence from trajectory construction while preserving PSNR/SSIM reconstruction performance across the evaluated sweeps. Do not turn this into a formal equivalence or unseen-domain theorem.

Not supported: all perceptual metrics are preserved or AFGM is a universal accuracy booster. Representative Liver LPIPS rises from 0.157043992 (MP) to 0.182352949 (AFGM), a +0.025308957 deterioration, even though PSNR/SSIM improve. PP-S/PP-A are not strictly better or worse in every metric.
''')
    doc('plgr/analysis.md','# PLGR internal audit\n\n'+provenance+'\n\nSources: `lgr/structural_sources.tsv`; cross-check: `lgr/final_lgr_table.tsv`. Legacy directory names only.\n\n'+'''
Liver 6 and Arm 5 identical paired sweeps across all five rows, equal-sweep means, MP, Full CGAS. No-global rows use the frozen local-aware fallback; global rows use path-aware adaptation. Consequently global contrasts measure the renderer within the complete adaptation protocol, not a pure inference-only effect at a fixed Gaussian population.

## Local formulation

Plane has no finite-slice conditioning and no global pathway; Fixed uses a constant effective slice width of 1 mm; Adaptive uses the retained depth-adaptive alternative and no global pathway. Plane is a true conditioning bypass, not merely an overlap-weight change.

Plane -> Fixed improves all three metrics in all 11 paired cases and is the dominant local-formulation gain. Fixed -> Adaptive raises Liver PSNR/SSIM but worsens LPIPS; on Arm it worsens all three metrics. Adaptive is an alternative comparison, not the final production method or a progressive addition.

## Global formulation

Fixed -> Fixed + Simple improves Arm PSNR/SSIM/LPIPS; on Liver it improves LPIPS but reduces PSNR AND SSIM. The latter trade-off must not be reduced to PSNR alone. Simple is the frozen scale-and-distance transmission model, not simply exponential pixel-depth attenuation.

Fixed + Simple -> Fixed + Full improves all three Arm aggregate metrics; Liver PSNR/SSIM/LPIPS all worsen slightly. This is a real-data benefit on the evaluated Arm subset with a small Liver trade-off, not an all-metric universal benefit. Full uses causal ray integration and independent attenuation, with unchanged production composition.

## Paired after-minus-before deltas

'''+md(pt)+'''
Win/tie/loss uses tolerance 1e-6 and reverses direction for LPIPS. These counts do not constitute significance tests. Five rows must NOT be narrated as a monotonic progression. Corrected Adaptive-only rows remain valid alternatives; old Adaptive+Full production rows are not substituted for Fixed+Full.
''')
    doc('cgas/analysis.md','# CGAS internal audit\n\n'+provenance+'\n\nSources: `cga/structural_sources.tsv`, `cga/event_totals.tsv`; cross-check: `cga/final_cga_table.tsv`. Legacy directory names only.\n\n'+common+'''

All rows use Fixed+Full PLGR and MP. GD = gradient densification plus conventional opacity pruning; RGD = residual-guided densification with NO pruning; RGD + OP adds opacity pruning; RGD + LP uses local-aware pruning; Full CGAS uses accumulated-path-contribution-aware pruning. These are strategy comparisons, not five isolated one-switch causal effects.

## Recomputed transitions

'''+md(ct)+'''
## Claim audit

- SUPPORTED descriptively: RGD retains substantially more capacity than GD. However GD -> RGD changes both densification and pruning, so do not attribute the entire final-population difference to residual guidance alone. Event totals separate spawned and pruned counts; RGD's final growth over the 1M initialization is much smaller than its relative increase over pruned GD.
- SUPPORTED: OP/LP sharply reduce the RGD population with narrow quality variation. Final populations and quality ranges are saved unrounded.
- NOT SUPPORTED as stated: “local-aware pruning is not consistently better than opacity-only in PSNR.” In these final two cases LP has HIGHER PSNR on both: Liver +0.0108706 dB; Arm +0.0050264 dB. These tiny single-seed differences do not establish a general accuracy benefit. LP also has slightly lower LPIPS on both.
- SUPPORTED: Full retains 43,672 additional primitives on Liver (+6.771493% relative to LP) and 22,418 on Arm (+3.245012%; 3.25% at two decimal places). This is a final population difference, not the number of distinct path-protected primitives.
- SUPPORTED: Full does not improve every quality metric. Relative to LP, Liver PSNR changes -0.0018494 dB while LPIPS improves -0.000497626; Arm PSNR improves +0.0386384 dB while LPIPS worsens +0.000701491.
- Safe interpretation: path-aware pruning selectively retains additional primitives while keeping reconstruction quality within a narrow range. Repeated path-protected candidate decisions in event_totals.tsv are not unique Gaussian counts.

#G is descriptive, with no bold/underline or better-direction. Raw integer counts and exact #G(k) are retained; manuscript display rounds #G(k) to integers, following the existing table.

Across all five variants, PSNR ranges are 0.0196663 dB (Liver) and 0.1082017 dB (Arm); LPIPS ranges are 0.001611883 and 0.000923958. GD -> RGD final population increases are 67.2754% / 50.4816%, whereas RGD growth over the common 1M initialization is only 5.0672% / 2.2586%. OP reduces RGD population by 38.7731% / 32.5371%; LP reduces it by 38.6165% / 32.4414%. This distinction prevents conflating suppressed pruning with newly spawned capacity.
''')
    errors=max(r['absolute_difference'] for r in CHECKS)
    lp=[r for r in effects if r['module']=='PLGR']
    audit='''# Final ablation table audit

## Scope and terminology

Four manuscript tables only: A Joint; B AFGM (representative + multi-sweep panels); C PLGR; D CGAS. No training/jobs, no default changes, no manuscript edits, no qualitative generation. Table labels exclusively use:

- AFGM — Acquisition-constrained Frame Geometry Model.
- PLGR — Physics-aware Local-Global Renderer.
- CGAS — Contribution-guided Gaussian Adaptation Strategy.

Old abbreviations appear only inside provenance paths, source mapping, or historical-audit explanation, never final table labels. Final renderer is Fixed 1-mm finite-slice + Full causal ray-integrated transmission; Adaptive is an explicitly marked local-only alternative.

## Authoritative data and verification

'''+provenance+'\n\n'+f'P2 equivalent source SHA-256: `{manifest["P2_source_sha256"]}`. All {len(CONFIGS)} unique archived configs pass row-specific semantic assertions. Current production resolves identically to frozen Liver/Arm full anchors for model/scene/loss/engine/metric/seed. Per-run case records: 16 Joint, 8 diagnostic, 32 multi-sweep, 55 PLGR, 10 CGAS; shared rows are legitimately reused, not independent runs. {len(CHECKS)} aggregate cells cross-checked; maximum numerical discrepancy versus frozen binary-float means = {errors}. Differences are arithmetic representation only. Source hashes are verified against the sealed manifest.\n\n'+'''Data-source hierarchy: frozen per-run final scalar values -> recomputed table means/sample standard deviations -> manuscript display. This audit does not re-evaluate models or recalculate original frame metrics. Complete exact run/config paths and hashes appear in each raw TSV.

## Verified PLGR factorial effects

'''+md(lp)+'''
The approximate +3.863 / +10.339 dB claims are verified. They are averages across four component backgrounds, not independent seeds.

## Display and ranking

Inspected existing exchange-table layouts and the frozen table-generation script, without editing them. Existing manuscript metrics are preserved: Joint/diagnostic/PLGR PSNR, SSIM, LPIPS; CGAS PSNR, LPIPS, #G(k). Quality displays 3 decimals; population displays integer thousands. Frozen generic six-decimal exports are not the manuscript precision. Canonical manuscript include path remains unconfirmed; this package provides new fragments rather than applying replacements.

Quality best/second marks use unrounded values, independently per dataset/metric/table. They describe numerical order, not practical/statistical importance or why a production default was selected. #G has no rank. Multi-sweep accuracy-preservation panel has no winner marks; sample std uses ddof=1. See rank_display_audit.tsv and rounding_ties.tsv. Equal-looking rounded cells are NOT promoted to joint winners. A caption note must state raw-value ranking; particularly CGAS LPIPS and Joint SSIM can show apparent ties.

## Important correction to expected science

Final RGD+LP PSNR exceeds RGD+OP on BOTH representatives (+0.0108706 / +0.0050264 dB). Therefore “not consistently better in PSNR” must NOT be copied. Replace with: observed differences are small and do not establish a general accuracy advantage.

Full vs LP population increases are verified: Liver 43,672 / 6.771493% (6.77% displayed); Arm 22,418 / 3.245012% (3.25% displayed).

For PLGR, Liver global additions trade SSIM as well as PSNR against perceptual quality; Full vs Simple slightly worsens all three Liver metrics, while Arm aggregate improves all three. Do not describe uniform gains or a monotonic five-row story.

## Stale assets — do not use as paper-final results

- The four `docs/production/experiment_results_io/tab_ablation_*.tex` exchange tables contain old values/labels. They were inspected only for layout/precision, not used as evidence; none was modified.
- Covariance-bug-era metrics; Adaptive+Full production Joint/diagnostic/CGAS results before the final Fixed refresh; the old four-row renderer table containing quadrature order as a structural row; old four-row adaptation table missing RGD+OP.
- Any old L/C/P joint labels copied without remapping: final column order is AFGM/PLGR/CGAS.
- Adaptive-only corrected alternatives in the PLGR table are VALID, not stale. Corrected older Fixed/Plane runs with certified P2 equivalence remain valid despite an earlier execution hash.
- Old joint qualitative figure is excluded from this package as requested. Existing originals remain untouched.

## Remaining caveats

No final numerical/configuration inconsistency was found. The expected LP-vs-OP claim requires textual correction as above. Representative rows do not support multi-seed significance. Spatial utility of path protection is supported by implementation/events, not by treating population differences as proof of unique protected identities. No additional experiment is requested or needed for this audit.
'''
    doc('FINAL_ABLATION_TABLE_AUDIT.md',audit)
    doc('README.md','''# Final ablation evidence package — 2026-09-16

Start with FINAL_ABLATION_TABLE_AUDIT.md; ALL_FOUR_TABLES.md contains every manuscript display value.

- joint/: eight AFGM/PLGR/CGAS combinations; raw per-run data, table, TeX body, paired effects and analysis.
- afgm/: four diagnostic rows and a six-row paired multi-sweep panel (one manuscript table); raw sources, sample statistics and interpretation.
- plgr/: five formulations, Liver 6 / Arm 5 equal-sweep means and paired transitions.
- cgas/: five strategies, representative metrics/populations, events and transition deltas.
- rank_display_audit.tsv: raw-value metric ranks and display values; population unranked.
- rounding_ties.tsv: apparent equalities after display rounding, not true ties.
- frozen_aggregate_verification.tsv: comparison with sealed aggregates.
- source_manifest.json / current_config_audit.json: input hashes and configuration checks.

New TeX files are standalone table BODY fragments, not edits to manuscript files. Each section's table.md names columns; Joint uses AFGM/PLGR/CGAS in that order. Quality metrics: 3 decimals; #G(k): integer. Bold best / underline second derives from unrounded values, except the multi-sweep preservation panel is deliberately unranked. std = across-sweep sample std, not across seeds.

No images, checkpoints, training, jobs, default changes or manuscript modifications. Internal per-run sources preserve exact old filenames only for provenance; final table labels use the new terminology.
''')


if __name__ == '__main__':
    main()
