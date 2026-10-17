from omegaconf import OmegaConf, DictConfig, ListConfig
from dataclasses import is_dataclass, fields
from typing import Any


def inspect_config_deep(obj: Any, name: str = "Root", indent: int = 0):
    """
    深度递归打印配置结构：包含 路径、数据类型 和 数值
    """
    spacing = "  " * indent
    
    # 获取当前对象的真实类型
    obj_type = type(obj)
    type_str = f"<{obj_type.__name__}>"
    
    # 处理 OmegaConf 容器或标准字典
    if isinstance(obj, (DictConfig, dict)):
        print(f"{spacing}📂 {name} {type_str}")
        items = obj.items() if isinstance(obj, dict) else obj.items()
        for k, v in items:
            inspect_config_deep(v, name=str(k), indent=indent + 1)
            
    # 处理 Dataclasses
    elif is_dataclass(obj):
        print(f"{spacing}📦 {name} {type_str}")
        for f in fields(obj):
            v = getattr(obj, f.name)
            inspect_config_deep(v, name=f.name, indent=indent + 1)
            
    # 处理 列表/序列
    elif isinstance(obj, (ListConfig, list, tuple)):
        print(f"{spacing}📜 {name} {type_str} (length: {len(obj)})")
        if len(obj) > 0:
            # 仅展示第一个元素作为类型参考，避免刷屏
            inspect_config_deep(obj[0], name="[0]", indent=indent + 1)
            if len(obj) > 1:
                print(f"{spacing}  ... (remaining {len(obj)-1} items)")
                
    # 处理 叶子节点 (数值、布尔、字符串等)
    else:
        # 特别标注：如果是可调用对象（方法/函数），用醒目的符号标出
        if callable(obj):
            print(f"{spacing}⚠️ {name}: {obj} {type_str} (CALLABLE!)")
        else:
            print(f"{spacing}🔹 {name}: {obj} {type_str}")
