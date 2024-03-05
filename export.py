import os
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
from utils.image_utils import depth_to_normal, unproject_depth_map

def export(dataset, pipe, iteration, downsample):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pcd = o3d.geometry.PointCloud()
    points = torch.empty((0,3)).cuda()
    colors = torch.empty((0,3)).cuda()
    normals = torch.empty((0,3)).cuda()
    
    print("\nUnprojecting")
    with torch.no_grad():
        for i in tqdm(range(len(cameras))):
            render_pkg = render(cameras[i], gaussians, pipe, background)
            image, depth = render_pkg["render"].permute(1,2,0), render_pkg["median_depth"]
            point = unproject_depth_map(depth, cameras[i])
            normal = depth_to_normal(depth, cameras[i])

            mask = (depth>0).squeeze()
            point = point[mask]
            color = image[mask]
            normal = normal[mask]

            indices = torch.randperm(len(point))[:int(len(point)/downsample)]

            points = torch.cat((points, point[indices]), dim=0)
            colors = torch.cat((colors, color[indices]), dim=0)
            normals = torch.cat((normals, normal[indices]), dim=0)  
    
    
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.cpu().numpy())

    print("\nSaving Point Cloud")
    o3d.io.write_point_cloud(os.path.join(args.model_path, 'pcd.ply'), pcd)

    print("\nComputing Poisson Mesh")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print("\nSaving Poisson Mesh")
    o3d.io.write_triangle_mesh(os.path.join(args.model_path, 'mesh.ply'), mesh)
    
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, default=7000)
    parser.add_argument('--downsample', type=int, default=10)
    args = parser.parse_args(sys.argv[1:])  
    print("Export " + args.model_path)

    export(lp.extract(args), pp.extract(args), args.iteration, args.downsample)

    print("\nExporting complete.")