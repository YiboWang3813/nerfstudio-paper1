import torch
import torch.nn.functional as F

from common.engine.base_runner import BaseRunner
from common.scene.factory import SceneFactory
from common.utils.metric_utils import psnr
from common.debug.debug_scope import DebugScope

from .gaussian_model import GaussianModel
from .render import render
from .density_control import DensityControlManager
from .hooks.gaussian_model import GaussianModelDebugHook
from .hooks.image import ImageDebugHook

class OfflineRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        self._build()

    def _build(self):
        # 1. 使用 Factory 组装 Scene
        self.scene = SceneFactory.create_from_args(self.args, self.device)

        # 2. 构建 Model 并利用 Scene 提供的 API 初始化
        self.model = GaussianModel(self.args, device=self.device)
        init_data = self.scene.initialize_gaussians(self.args.init_num)
        self.model.init_from_scene(init_data)
        
        self.model.setup_optimizer()
        self.model.setup_lr_scheduler()

        # 3. 密度控制
        self.dc = DensityControlManager(self.args)
        self.dc.on_model_initialized(self.model)

        # 4. 注册并加载 Checkpoint
        self._ckpt_items = [("gaussian", self.model), ("dc", self.dc)]
        self.start_iter = self._load_checkpoint()

    def _train_one_step(self, iteration):
        # 数据获取
        world = self.scene.step(sample_gaussians=False, evaluate=False)
        curr_idx = world["current_index"]

        # 前向 Hook
        if DebugScope.enabled(): GaussianModelDebugHook.on_forward(self.model)

        # 渲染与损失
        pred, gt, loss = self._process_frame(curr_idx, is_train=True)

        # 优化
        self.model.zero_grad()
        loss["total"].backward()
        self.model.optimizer.step()

        # 后向 Hook
        if DebugScope.enabled():
            ImageDebugHook.on_images(pred, gt)
            GaussianModelDebugHook.on_backward(self.model)

        # 密度控制步末记录
        self.dc.on_step_end(self.model)

        # 密度控制触发
        if iteration % self.args.dc_freq == 0:
            self.dc.prune_and_densify(self.model, iteration)

        return {"loss": loss, "pred": pred, "gt": gt}

    def _process_frame(self, index, is_train=True):
        H, W = self.args.image_height, self.args.image_width
        p_pos, p_int = self.scene.build_pixels(index=index)
        gt_image = p_int.view(H, W)

        # 调用已模块化的 render
        pred_image = render(self.args, p_pos, self.model, index)
        
        loss_dict = self._compute_loss(pred_image, gt_image) if is_train else None
        return pred_image, gt_image, loss_dict

    def _compute_loss(self, pred, gt):
        l1 = (1.0 - self.args.lambda_ssim) * self.l1_loss(pred, gt)
        ssim = self.args.lambda_ssim * self.ssim_loss(pred, gt)
        # S2S 特有约束：Scale 惩罚
        scale_reg = 0.1 * torch.mean(F.relu(self.model.scales - 1.0)**2)
        return {"total": l1 + ssim + scale_reg, "l1": l1, "ssim": ssim, "scale": scale_reg}

    @torch.no_grad()
    def _test(self, iteration):
        test_indices = self.scene.scheduler.get_test_indices()
        l1_sum, psnr_sum = 0.0, 0.0
        for idx in test_indices:
            pred, gt, _ = self._process_frame(int(idx), is_train=False)
            l1_sum += self.l1_loss(pred, gt).item()
            psnr_sum += psnr(pred, gt).item()
        
        n = len(test_indices)
        self.monitor.update(iteration, **{"test/metric/l1": l1_sum/n, "test/metric/psnr": psnr_sum/n})
        self.logger.log(self.monitor.latest(), mode="test")