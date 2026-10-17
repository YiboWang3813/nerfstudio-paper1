"""Run installed nature-figure PDF QA without touching experiments."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'docs/production/final_hyperparameter_figures_20260916'
SKILL = Path.home() / '.codex/skills/nature-figure/scripts'


def main():
    reports = []
    targets = ['H1_AFGM/H1', 'H1_AFGM/H1_candidate_B', 'H2_PLGR/H2',
               'H2_PLGR/H2_lpips_review', 'H3_CGAS/H3', 'H4_CGAS_population/H4']
    for target in targets:
        base = OUT / target
        font = subprocess.run([sys.executable, str(SKILL / 'audit_pdf_text.py'),
                               str(base) + '.pdf', '--min-pt', '5', '--json'],
                              capture_output=True, text=True)
        Path(str(base) + '.font-audit.json').write_text(font.stdout)
        collision = subprocess.run([sys.executable, str(SKILL / 'audit_figure_collisions.py'),
                                    str(base) + '.pdf', '--json-out', str(base) + '.collision-audit.json',
                                    '--overlay-pdf', str(base) + '.collision-overlay.pdf'],
                                   capture_output=True, text=True)
        reports.append(dict(figure=target, font_exit=font.returncode,
                            collision_exit=collision.returncode, collision_summary=collision.stdout))
        print(target, font.returncode, collision.returncode)
        if collision.returncode:
            print(collision.stdout)
    (OUT / 'skill_pdf_qa.json').write_text(json.dumps(reports, indent=2))
    return int(any(r['font_exit'] or r['collision_exit'] for r in reports))


if __name__ == '__main__':
    sys.exit(main())
