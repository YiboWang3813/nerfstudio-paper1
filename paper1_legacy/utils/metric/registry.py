"""
Metric 注册表：与 utils/loss/registry 对称，供 methods/.../computers 统一装配。
"""
METRIC_CRITERION_REGISTRY = {}


def register_metric_criterion(name: str):
    def decorator(cls):
        METRIC_CRITERION_REGISTRY[name] = cls
        return cls
    return decorator
