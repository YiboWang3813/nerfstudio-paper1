import os 
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import imageio
from torch.nn import MSELoss
from pytorch_msssim import SSIM


TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cyclically_shift_dims_left(tensor: torch.Tensor):
    order = tuple(range(1, len(tensor.shape))) + (0, )
    return tensor.permute(order)

def add_a_leading_zero(tensor: torch.Tensor, dim=0):
    dim %= len(tensor.shape)
    zero = torch.zeros_like(torch.index_select(tensor, dim=dim, index=torch.tensor([0], dtype=torch.long)))
    return torch.concat([zero, tensor], dim=dim)

def add_a_leading_one(tensor: torch.Tensor, dim=0):
    dim %= len(tensor.shape)
    one = torch.ones_like(torch.index_select(tensor, dim=dim, index=torch.tensor([0], dtype=torch.long)))
    return torch.concat([one, tensor], dim=dim)

def repeat_last_element(tensor: torch.Tensor, dim=0):
    # along the <dim> axis, find the last element (row or col) in tensor, then concatenate it to the end of tensor along the <dim> axis
    dim %= len(tensor.shape)
    last_element = torch.index_select(tensor, dim=dim, index=torch.tensor([tensor.shape[dim] - 1], dtype=torch.long))
    return torch.concat([tensor, last_element], dim=dim)

def plot_points(points, ref=None):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.dpi'] = 300
    points = points.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if ref is not None:
        ref = ref.reshape(-1, 3)
        ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], c='r', marker='x')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.savefig('points.png')

def sample_bernoulli(probabilities_yielding_one):
    probabilities_yielding_zero = 1 - probabilities_yielding_one # (num_points_per_ray, num_rays)
    logits = torch.logit(torch.stack([probabilities_yielding_one, probabilities_yielding_zero], dim=-1), eps=1e-4) # (num_points_per_ray, num_rays)
    return torch.nn.functional.gumbel_softmax(logits, tau=1e-1, hard=True)[..., 0] # (num_points_per_ray, num_rays)

def pose_vector_to_matrix(poses):
    rot_mats = R.from_quat(poses[:, -4:]).as_matrix()
    tfs = np.concatenate([np.concatenate([rot_mats, poses[:, :3].reshape(poses[:, :3].shape + (1,))], axis=-1), np.broadcast_to(np.array([0, 0, 0, 1.]).reshape(1, 1, 4), (len(poses), 1, 4))], axis=1)
    return tfs

def set_torch_default_device_to_cuda_if_available():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

def save_side_by_side_image(img1: np.ndarray, img2: np.ndarray, save_path):
    if not img1.shape == img2.shape:
        raise ValueError(f'The two images should have the same shape, got {img1.shape} and {img2.shape} instead.')
    if len(img1.shape) < 2:
        raise ValueError(f'Inputs must be at least 2D, got shape {img1.shape} instead.')
    if img1.dtype == np.float32 and img2.dtype == np.float32:
        side_by_side_image = np.concatenate([ (img1 * 255).astype(np.uint8), (img2 * 255).astype(np.uint8) ], axis=1)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.imsave(save_path, side_by_side_image)
    else:
        raise NotImplementedError

def loss_function(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global ssim_global_variable, l2_global_variable
    if not 'ssim_global_variable' in globals():
        ssim_global_variable = SSIM(data_range=1.0, size_average=True, channel=1)
    if not 'l2_global_variable' in globals():
        l2_global_variable = MSELoss(reduction='mean')
    weight_ssim = 1 - 1e-2
    weight_l2 = 1 - weight_ssim

    loss_ssim = 1 - ssim_global_variable(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))
    loss_l2 = l2_global_variable(pred, gt)
    loss = weight_ssim * loss_ssim + weight_l2 * loss_l2
    return loss, loss_ssim, loss_l2

if __name__ == '__main__':
    pass