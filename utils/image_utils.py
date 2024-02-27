#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import matplotlib.pyplot as plt

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def depth_to_normal(depth_map, camera):
    # depth_map = depth_map[None, ...]

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()
    

    grad_x = torch.nn.functional.conv2d(depth_map, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(depth_map, sobel_y, padding=1)

    dz = torch.ones_like(grad_x)
    normal_map = torch.cat((grad_x, grad_y, -dz), 0)
    norm = torch.norm(normal_map, p=2, dim=0, keepdim=True)
    normal_map = normal_map / norm

    return normal_map

def unproject_depth_map(camera, depth_map):
    height, width = depth_map.shape
    x = torch.linspace(0, width - 1, width).cuda()
    y = torch.linspace(0, height - 1, height).cuda()
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Reshape the depth map and grid to N x 1
    depth_flat = depth_map.reshape(-1)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)

    # Normalize pixel coordinates to [-1, 1]
    X_norm = (X_flat / (width - 1)) * 2 - 1
    Y_norm = (Y_flat / (height - 1)) * 2 - 1

    # Create homogeneous coordinates in the camera space
    points_camera = torch.stack([X_norm, Y_norm, depth_flat], dim=-1)
    # points_camera = points_camera.view((height,width,3))
    

    K_matrix = camera.projection_matrix
    # parse out f1, f2 from K_matrix
    f1 = K_matrix[2, 2]
    f2 = K_matrix[3, 2]
    # get the scaled depth
    sdepth = (f1 * points_camera[..., 2:3] + f2) / points_camera[..., 2:3]
    # concatenate xy + scaled depth
    points_camera = torch.cat((points_camera[..., 0:2], sdepth), dim=-1)


    points_camera = points_camera.view((height,width,3))
    points_camera = torch.cat([points_camera, torch.ones_like(points_camera[:, :, :1])], dim=-1)  
    points_world = torch.matmul(points_camera, camera.full_proj_transform.inverse())

    # Discard the homogeneous coordinate
    points_world = points_world[:, :, :3] / points_world[:, :, 3:]
    
    return points_world

def colormap(map, cmap="magma"):
    colors = plt.cm.get_cmap(cmap).colors
    start_color = torch.tensor(colors[0]).view(3, 1, 1).to(map.device)
    end_color = torch.tensor(colors[-1]).view(3, 1, 1).to(map.device)
    return (1 - map) * start_color + map * end_color