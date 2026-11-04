"""Produce config and four-dataset signatures from one legacy source tree."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest_array(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from configs.setup import load_hydra_config
    from omegaconf import OmegaConf
    import torch
    from scene.factory import SceneFactory
    configs = {}
    for path in sorted((root/'configs_hydra/experiment').rglob('*.yaml')):
        alias = str(path.relative_to(root/'configs_hydra/experiment').with_suffix(''))
        try:
            cfg = load_hydra_config(alias)
            text = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
            configs[alias] = {'sha256': hashlib.sha256(text.encode()).hexdigest()}
        except Exception as exc:
            configs[alias] = {'error_type': type(exc).__name__, 'error': str(exc).replace(str(root), '<ROOT>')}
    datasets = {}
    for scene in ('liver', 'spine_phantom', 'tdsc_abus_2023', 'tus_rec_2024_qc'):
        cfg = load_hydra_config('production/ablations/ours/paper_semantic_freeze', [
            'scene='+scene, 'scene.loading.frame_end=8',
            'scene.loading.holdout_trim_each=1', 'scene.loading.holdout_num_samples=2',
            'system.device=cpu'])
        container = SceneFactory.build_scene(cfg, torch.device('cpu'))
        datum = container.train_container.get(0)
        datasets[scene] = {'frame_indices': container.train_container.frame_indices.tolist(),
                           'train_indices': container.train_indices, 'test_indices': container.test_indices,
                           'image_shape': list(datum.image.shape), 'image_sha256': digest_array(datum.image),
                           'pose_sha256': digest_array(datum.pose),
                           'uses_file_poses_for_geometry': container.train_container.uses_file_poses_for_geometry,
                           'spacing': [container.dataset_meta.delta_lateral_mm, container.dataset_meta.delta_axial_mm]}
    report = {'configs': configs, 'datasets': datasets, 'python': sys.version, 'torch': torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print('REPLAY_SIGNATURES', len(configs), len(datasets), flush=True)


if __name__ == '__main__':
    main()
