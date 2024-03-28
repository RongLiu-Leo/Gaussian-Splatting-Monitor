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
from utils import init_logger, NetworkGUI
from pathlib import Path
import torch
import subprocess

def main(config_path):

    # parse cfg
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    name = cfg.get("name")

    render_settings = cfg.get("render_settings")
    save_path = Path(render_settings.get("save_path"))
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(render_settings.get("model_path"))

    # init logger
    logger = init_logger(name, save_path)

    # init repr
    repr = GaussianRepr(cfg = cfg.get('repr'), logger = logger)
    repr.load_ply(model_path)

    # init network gui
    networkGui = NetworkGUI(cfg = cfg.get('networkGui'), logger = logger)

    # init renderer
    renderer = DiffRasterizerRenderer(cfg = cfg.get('renderer'), logger = logger)
    
    # init camera data(use exist camera as example)
    data = ColmapData(cfg = cfg.get('data'), logger = logger)

    # view!
    while True:
        with torch.no_grad():
            networkGui.process(renderer, repr, data)

if __name__ == "__main__":
    try:
        subprocess.Popen("./SIBR_viewers/install/bin/SIBR_remoteGaussian_app_rwdi.exe")
    except FileNotFoundError:
        print("SIBR viewer not found. Please install it through the SIBR_viewers repository.")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)