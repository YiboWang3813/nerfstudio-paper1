import numpy as np
import torch
import os

# 修改为你的路径
sweep_dir = "/home/boywang25/ultrasound-scene-reconstruction/data/synthetic_liver/l1" 
poses_path = os.path.join(sweep_dir, "poses.npy")

if os.path.exists(poses_path):
    poses = np.load(poses_path)
    
    # 提取位移 (Translation)
    # 假设 poses 是 (N, 4, 4)
    trans = poses[:, :3, 3] 
    
    # 计算相邻帧的欧氏距离
    diffs = trans[1:] - trans[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    
    avg_dist = np.mean(dists)
    total_len = np.sum(dists)
    
    print(f"--- Data Analysis ---")
    print(f"Total Frames: {len(poses)}")
    print(f"Average Delta (Step Size): {avg_dist:.5f} mm") # <--- 记下这个数！
    print(f"Total Sweep Length: {total_len:.2f} mm")
    
    # 检查是不是扇扫 (Sector Sweep)
    if avg_dist < 0.01:
        print("[WARNING] Delta is near zero! This might be a pure rotation sweep (Sector).")
else:
    print("Poses file not found.")