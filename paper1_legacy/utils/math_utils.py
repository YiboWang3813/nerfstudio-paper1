import torch 


def sigmoid_inv(x):
    return torch.log(x/(1-x))

def sigmoid_with_max(raw, max_val): 
    return torch.sigmoid(raw) * max_val 

def inverse_sigmoid_with_max(act, max_val, eps=1e-8): 
    # 先还原到0-1 
    act = torch.clamp(act / max_val, eps, 1 - eps) 
    return torch.log(act / (1 - act))    

def softplus_inv(y, eps=1e-6):
    y = torch.clamp(y, min=eps)
    return torch.log(torch.expm1(y))
