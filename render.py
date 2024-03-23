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

import yaml
import argparse
from dataloader import ColmapData
from model import GaussianRepr, DiffRasterizerRenderer
from utils import setup_logging, init_logger
from pathlib import Path
from PIL import Image
def main(config_path):

    # parse cfg
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    name = cfg.get("name")
    render_settings = cfg.get("render_settings")
    save_path = Path(render_settings.get("save_path"))
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(render_settings.get("model_path"))
    render_modes = render_settings.get("render_modes")

    # init logger
    setup_logging(cfg.get('logger'), save_path)
    logger = init_logger(name)

    # init repr
    repr = GaussianRepr(cfg = cfg.get('repr'), logger = logger)
    repr.load_ply(model_path)

    # init renderer
    renderer = DiffRasterizerRenderer(cfg = cfg.get('renderer'), logger = logger)
    
    # init camera data(use exist camera as example)
    data = ColmapData(cfg = cfg.get('data'), logger = logger)
    cameras = [pair.camera for pair in data.get_train_pair_list()][:10]

    # render!
    for cam_idx, camera in enumerate(cameras):
        for render_mode in render_modes:
            img = renderer.render_img(repr = repr, camera = camera, render_mode = render_mode)
            img_path = save_path /f"{cam_idx}_{render_mode}.png"
            img = img.detach().cpu().numpy()  # Convert tensor to numpy array
            img = (img * 255).clip(0, 255).astype('uint8')
            image = Image.fromarray(img.transpose(1, 2, 0))
            image.save(img_path)  # Save the image

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)