import argparse

def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("S2S Specific")
    
    # S2S 特有的 Loss 权重
    group.add_argument("--lambda_ssim", type=float, default=0.2)

    # S2S 默认的学习率分布 (可以覆盖 base)
    group.add_argument("--means_lr", type=float, default=5e-5)
    group.add_argument("--means_lr_init", type=float, default=5e-5)
    group.add_argument("--means_lr_final", type=float, default=1e-5)
    group.add_argument("--means_lr_delay_mult", type=float, default=0.01)
    group.add_argument("--means_lr_max_steps", type=int, default=30_000)
    group.add_argument("--scales_lr", type=float, default=1e-5)
    group.add_argument("--quats_lr", type=float, default=2e-4)
    group.add_argument("--opacities_lr", type=float, default=5e-4)
    group.add_argument("--intensities_lr", type=float, default=1e-3)

    # S2S 特有的密度控制参数
    group.add_argument("--dc_freq", type=int, default=200_000)
    group.add_argument("--dc_prune_ratio", type=float, default=0.05) 
    group.add_argument("--dc_refill_ratio", type=float, default=0.9) 
    group.add_argument("--dc_min_pop", type=int, default=30_000) 
    group.add_argument("--dc_opa_ths_init", type=float, default=0.05) 
    group.add_argument("--dc_opa_ths_final", type=float, default=0.08) 

    return parser

def process_args(args):
    """S2S 特有后处理 (如有)"""
    return args