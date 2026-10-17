import torch

class SceneBBoxMeta:
    """
    [Data Class]
    纯数据类：存储场景的物理边界状态，提供只读的几何统计属性。
    """
    __slots__ = ("grid_min", "grid_max", "tile_size", "grid_size")

    def __init__(self, grid_min, grid_max, tile_size, grid_size):
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.tile_size = tile_size
        self.grid_size = grid_size

    # === 几何统计属性 (只读) ===
    @property
    def scene_span(self) -> torch.Tensor:
        """(x, y, z) 物理长度"""
        return self.grid_max - self.grid_min

    @property
    def scene_radius(self) -> float:
        """场景半径 (用于 LR Scaling)"""
        return torch.norm(self.scene_span).item() / 2.0

    @property
    def scene_extent(self) -> float:
        """场景最大边长"""
        return self.scene_span.max().item()

    def get_suggested_position_lr(self, base_lr: float = 0.00016) -> float:
        """物理自适应学习率"""
        return base_lr * self.scene_radius

    def to(self, device: torch.device):
        self.grid_min = self.grid_min.to(device)
        self.grid_max = self.grid_max.to(device)
        self.tile_size = self.tile_size.to(device)
        self.grid_size = self.grid_size.to(device)
        return self
    
    def __repr__(self):
        return (f"<SceneBBoxMeta radius={self.scene_radius:.2f} "
                f"extent={self.scene_extent:.2f} "
                f"grid_size={self.grid_size.tolist()}>")


class SceneBBoxComputer:
    """
    [Logic Class]
    计算类：持有超参数 (tile_size, margin)，负责生成和更新 Meta。
    """
    def __init__(self, tile_size: float, margin_ratio: float = 0.05):
        self.tile_size_val = tile_size
        self.margin_ratio = margin_ratio

    @torch.no_grad()
    def compute_init(self, points: torch.Tensor) -> SceneBBoxMeta:
        """从点云初始化 Meta"""
        device = points.device
        tile_size = torch.tensor([self.tile_size_val] * 3, device=device)

        pmin = points.min(dim=0).values
        pmax = points.max(dim=0).values
        span = pmax - pmin

        # 施加 Margin
        raw_min = pmin - span * self.margin_ratio
        raw_max = pmax + span * self.margin_ratio

        # Tile 对齐
        grid_min = torch.floor(raw_min / tile_size) * tile_size
        grid_max = torch.ceil(raw_max / tile_size) * tile_size
        grid_size = torch.ceil((grid_max - grid_min) / tile_size).to(torch.int32)

        return SceneBBoxMeta(
            grid_min=grid_min.contiguous(),
            grid_max=grid_max.contiguous(),
            tile_size=tile_size.contiguous(),
            grid_size=grid_size.contiguous()
        )

    @torch.no_grad()
    def update(self, meta: SceneBBoxMeta, new_points: torch.Tensor) -> bool:
        """
        更新现有的 Meta。
        如果发生更新，直接修改 meta 对象内部属性，并返回 True。
        """
        device = meta.grid_min.device
        new_pos = new_points.to(device)
        
        pmin = new_pos.min(dim=0).values
        pmax = new_pos.max(dim=0).values

        needs_update_min = (pmin < meta.grid_min).any()
        needs_update_max = (pmax > meta.grid_max).any()

        if needs_update_min or needs_update_max:
            # 扩展策略：取 (当前边界) 和 (新点扩展边界) 的并集
            # 这里 margin 稍微给小一点用于增量更新，或者复用 init 的 margin
            margin = self.margin_ratio
            
            # 计算新点需要的理想边界
            raw_min = torch.min(meta.grid_min, pmin - (pmax - pmin) * margin)
            raw_max = torch.max(meta.grid_max, pmax + (pmax - pmin) * margin)

            # 重新对齐写入 Meta
            meta.grid_min = torch.floor(raw_min / meta.tile_size) * meta.tile_size
            meta.grid_max = torch.ceil(raw_max / meta.tile_size) * meta.tile_size
            
            # 更新 size
            span = meta.grid_max - meta.grid_min
            meta.grid_size = torch.ceil(span / meta.tile_size).to(torch.int32)
            
            return True 
            
        return False