STATE_REGISTRY = {}
CALLBACK_REGISTRY = {}

def register_state(name):
    def decorator(cls):
        STATE_REGISTRY[name] = cls
        return cls
    return decorator

def register_callback(name):
    def decorator(cls):
        CALLBACK_REGISTRY[name] = cls
        return cls
    return decorator