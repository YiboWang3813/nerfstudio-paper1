"""Native baseline model/Trainer contracts, including their actual CUDA paths."""
import copy
import pytest
import torch

from nerfstudio.configs.ultrasound_baselines import GAUSSIAN_METHODS,baseline_config
from nerfstudio.configs.ultrasound_baselines import field_config


@pytest.mark.parametrize('method',['nerfus','ultranerf'])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA execution')
def test_native_acoustic_field_training(method,manifest,tmp_path):
    config=field_config(method)
    config.pipeline.datamanager.dataparser.data=manifest
    algorithm=config.pipeline.model.algorithm
    algorithm.field.width=16;algorithm.field.depth=2;algorithm.field.skip_layer=1
    algorithm.ray.train_patch_height=12;algorithm.ray.train_patch_width=12
    config.output_dir=tmp_path/'runs';config.experiment_name=method;config.timestamp='test'
    config.logging.local_writer.enable=False;config.max_num_iterations=3
    config.steps_per_eval_batch=config.steps_per_eval_image=config.steps_per_eval_all_images=0
    trainer=config.setup();trainer.setup();trainer.train()
    metrics,_=trainer.pipeline.get_eval_image_metrics_and_images(3)
    assert torch.isfinite(torch.tensor(metrics['psnr']))
    assert all(torch.isfinite(p).all() for p in trainer.pipeline.model.parameters())


@pytest.mark.parametrize('method',GAUSSIAN_METHODS)
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires actual CUDA execution')
def test_native_baseline_train_checkpoint_resume(method,manifest,tmp_path):
    config=baseline_config(method)
    config.pipeline.datamanager.dataparser.data=manifest
    config.pipeline.model.num_gaussians=32
    if method=='slice_adapted_3dgs':config.pipeline.model.algorithm.render.backend='cuda'
    control=config.pipeline.model.algorithm.offline.density_control
    control.timing.start_iter=1;control.timing.end_iter=4;control.timing.freq=2
    control.enable.densify=True;control.enable.prune=False
    config.output_dir=tmp_path/'runs';config.experiment_name=method;config.timestamp='first'
    config.logging.local_writer.enable=False
    config.max_num_iterations=3
    config.steps_per_eval_batch=config.steps_per_eval_image=config.steps_per_eval_all_images=0
    trainer=config.setup();trainer.setup();trainer.train()
    assert all(torch.isfinite(p).all() for p in trainer.pipeline.model.parameters())
    checkpoint=next(trainer.checkpoint_dir.glob('*.ckpt'))
    loaded=copy.deepcopy(config);loaded.timestamp='restored';loaded.load_checkpoint=checkpoint
    restored=loaded.setup();restored.setup()
    for key,value in trainer.pipeline.model.gaussians.items():
        torch.testing.assert_close(value,restored.pipeline.model.gaussians[key])
    metrics,_=restored.pipeline.get_eval_image_metrics_and_images(3)
    assert torch.isfinite(torch.tensor(metrics['psnr']))
