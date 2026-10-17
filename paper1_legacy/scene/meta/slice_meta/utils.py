# dataset/meta/utils.py
import torch


def matrix_to_quaternion(R: torch.Tensor):
    """
    Robust matrix to quaternion (w, x, y, z)
    Handles all cases ensuring numerical stability.
    R: (3, 3)
    """
    if R.dim() == 2: R = R.unsqueeze(0) # (1, 3, 3)
    
    # 既然是一帧一帧算，Batch size = 1，我们可以写得直白点
    R00, R01, R02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    R10, R11, R12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    R20, R21, R22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    
    trace = R00 + R11 + R22
    q = torch.zeros((R.shape[0], 4), device=R.device)

    # Case 1: Trace > 0
    if trace > 0:
        S = torch.sqrt(trace + 1.0) * 2
        q[:, 0] = 0.25 * S
        q[:, 1] = (R21 - R12) / S
        q[:, 2] = (R02 - R20) / S
        q[:, 3] = (R10 - R01) / S
    # Case 2: R00 is max
    elif (R00 > R11) and (R00 > R22):
        S = torch.sqrt(1.0 + R00 - R11 - R22) * 2
        q[:, 0] = (R21 - R12) / S
        q[:, 1] = 0.25 * S
        q[:, 2] = (R01 + R10) / S
        q[:, 3] = (R02 + R20) / S
    # Case 3: R11 is max
    elif R11 > R22:
        S = torch.sqrt(1.0 + R11 - R00 - R22) * 2
        q[:, 0] = (R02 - R20) / S
        q[:, 1] = (R01 + R10) / S
        q[:, 2] = 0.25 * S
        q[:, 3] = (R12 + R21) / S
    # Case 4: R22 is max
    else:
        S = torch.sqrt(1.0 + R22 - R00 - R11) * 2
        q[:, 0] = (R10 - R01) / S
        q[:, 1] = (R02 + R20) / S
        q[:, 2] = (R12 + R21) / S
        q[:, 3] = 0.25 * S
        
    return q.squeeze(0) # (4,)


def quat_multiply(q1, q2):
    """ Hamilton product: q1 * q2. Shape: (N, 4) """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)