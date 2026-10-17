import argparse

def add_args(parser: argparse.ArgumentParser):
    """Ultra-NeRF 特定参数定义"""
    group = parser.add_argument_group("Ultra-NeRF Specific")

    # ================= Network Architecture =================
    group.add_argument("--net_depth", type=int, default=8, help="Layers in network")
    group.add_argument("--net_width", type=int, default=256, help="Channels per layer")
    group.add_argument("--use_viewdirs", type=bool, default=False, help="Enable view dependence (not used by paper-aligned isotropic field)")
    
    # Positional Encoding
    group.add_argument("--multires", type=int, default=10, help="Log2 of max freq for positional encoding (3D location)")
    group.add_argument("--multires_views", type=int, default=4, help="Log2 of max freq for positional encoding (2D direction)")

    # ================= Physics & Rendering =================
    # PSF (Point Spread Function) Sigma values for anisotropic gaussian kernel
    # 通常超声图像在横向(Lateral)比轴向(Axial)更模糊
    group.add_argument("--psf_sigma_lateral", type=float, default=1.0, help="PSF sigma for lateral direction")
    group.add_argument("--psf_sigma_axial", type=float, default=0.5, help="PSF sigma for axial direction")

    # ================= Optimization =================
    group.add_argument("--lr_init", type=float, default=5e-4, help="Initial learning rate for Adam")
    # BaseArgs 里虽然有 iterations，但 Ultra-NeRF 论文通常使用不同的训练步数
    group.add_argument("--n_iters", type=int, default=20_000, help="Total training iterations for Scheduler decay calculation")

    # ================= Loss Weights =================
    # Ultra-NeRF 论文强调 SSIM 的重要性
    group.add_argument("--lambda_l1", type=float, default=1.0, help="Weight for L1 pixel loss")
    group.add_argument("--lambda_ssim", type=float, default=0.5, help="Weight for SSIM structural loss")

    return parser

def process_args(args):
    # 确保 n_iters 逻辑一致性 (如果用户在命令行覆盖了 global iterations)
    if args.iterations != 30_000: # 如果用户显式修改了 base 的 iterations
        args.n_iters = args.iterations

    msg = "[Ultra-NeRF]: "
    msg += f"Network D={args.net_depth}, W={args.net_width}, ViewDirs={args.use_viewdirs}. "
    msg += f"PSF Sigma=(Lat:{args.psf_sigma_lateral}, Ax:{args.psf_sigma_axial}).\n"
    msg += f"Optimization: LR={args.lr_init} decay over {args.n_iters} steps. "
    msg += f"Loss: L1={args.lambda_l1}, SSIM={args.lambda_ssim}."
    print(msg)

    return args
