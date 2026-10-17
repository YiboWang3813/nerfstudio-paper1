
LOSS_CRITERION_REGISTRY = {}

def register_loss_criterion(name):
    def decorator(cls):
        LOSS_CRITERION_REGISTRY[name] = cls
        return cls
    return decorator