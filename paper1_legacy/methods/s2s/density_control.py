import os
import torch
import torch.nn.functional as F

class DensityControlManager:
    """
    Optimized Strict DC v2:
    1. 动态阈值：随训练进程提高剪枝门槛，后期清理更狠。
    2. 非对称重采样：维持 0.7 补回率，促使点数精简化。
    3. 数值稳定性：强化 NaN 检查。
    """
    def __init__(self, args):
        self.prune_ratio = args.dc_prune_ratio
        self.refill_ratio = args.dc_refill_ratio
        self.min_population = args.dc_min_pop
        self.opacity_threshold_init = args.dc_opa_ths_init
        self.opacity_threshold_final = args.dc_opa_ths_final
        self.max_steps = args.iterations 

    def on_model_initialized(self, model): pass
    def on_step_end(self, model): pass

    @torch.no_grad()
    def prune_and_densify(self, model, step):
        device = model.means.device
        current_pop = model.means.shape[0]
        
        # 计算当前步数下的动态透明度阈值 (线性增长)
        progress = min(1.0, step / self.max_steps)
        current_opacity_thresh = self.opacity_threshold_init + \
                                (self.opacity_threshold_final - self.opacity_threshold_init) * progress

        # 1. 组合判定剪枝
        measurement = torch.abs(model.opacities * model.intensities)
        eps = torch.quantile(measurement, self.prune_ratio)
        mask_pct = (measurement >= eps)
        
        # 使用动态阈值进行剪枝
        mask_abs = (model.opacities > current_opacity_thresh)
        
        keep_mask = (mask_pct & mask_abs).flatten()
        num_pruned = (~keep_mask).sum().item()

        model.prune(keep_mask)

        # 2. 计算重采样数量
        num_to_refill = int(num_pruned * self.refill_ratio)
        if (model.means.shape[0] + num_to_refill) < self.min_population:
            num_to_refill = max(0, self.min_population - model.means.shape[0])

        if num_to_refill > 0:
            # 3. 稳健的包围盒采样
            bb_min = torch.quantile(model.means, 0.01, dim=0)
            bb_max = torch.quantile(model.means, 0.99, dim=0)

            new_latents = {}
            new_latents["means"] = bb_min + torch.rand((num_to_refill, 3), device=device) * (bb_max - bb_min)

            # 4. 初始化
            for name, cfg in model.param_cfg.items():
                if name == "means": continue
                init_val = cfg["init"](num_to_refill, device=device)
                
                if name == "scales":
                    # 初始 scale 略微减小，配合正则项，防止后期失控
                    init_val = init_val * (1.0 + (torch.rand_like(init_val) - 0.5) * 0.2)
                if name == "quats":
                    init_val = F.normalize(torch.randn((num_to_refill, 4), device=device), dim=-1)
                
                new_latents[name] = cfg["inverse"](init_val)

            model.concat(new_latents)

        # 5. 清理
        for name, cfg in model.param_cfg.items():
            p = getattr(model, cfg["param"])
            p.data.copy_(torch.nan_to_num(p.data, nan=0.0))

        final_pop = model.means.shape[0]
        print(f"[DC-Dynamic] Step: {step} | Thresh: {current_opacity_thresh:.3f} | "
              f"Pop: {current_pop}->{final_pop} | Pruned: {num_pruned}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({}, path)

    def load(self, path: str, model): pass