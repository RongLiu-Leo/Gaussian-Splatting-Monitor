import torch
from random import randint
from model.base import BaseModule
from pathlib import Path
import json
from utils import ProgressBar
class BaseTrainer(BaseModule):
    def __init__(self, cfg, logger, data, repr, loss, renderer, paramOptim, structOptim,
                 result_path, ckpt_path):
        super().__init__(cfg, logger)
        self.data = data
        self.repr = repr
        self.loss = loss
        self.paramOptim = paramOptim
        self.renderer = renderer
        self.structOptim = structOptim
        self.result_path = Path(result_path)
        self.ckpt_path = Path(ckpt_path)

        self.progress_bar = ProgressBar(self.iterations + 1)

    def init_save_results(self):
        # Save point cloud data
        input_ply_path = self.result_path / "input.ply"
        with open(self.data.ply_path, 'rb') as src_file, open(input_ply_path , 'wb') as dest_file:
            dest_file.write(src_file.read())

        # Save camera data
        camera_path = self.result_path /  "cameras.json"
        json_cams = [camera_image_pair.json for camera_image_pair in self.data.get_train_pair_list() + self.data.get_test_pair_list()]
        with open(camera_path, 'w') as file:
            json.dump(json_cams, file)
        
        # Save config
        is_white_background = self.renderer.background_color == [255,255,255]
        namespace_str = f"Namespace(data_device='cuda', \
                                eval={self.data.eval}, \
                                images='images', \
                                model_path='{str(self.result_path)}', \
                                resolution={self.data.resolution}, \
                                sh_degree={self.repr.max_sh_degree}, \
                                source_path='{self.data.source_path}', \
                                white_background={is_white_background})"
        with open(self.result_path / "cfg_args", 'w') as cfg_log_f:
            cfg_log_f.write(namespace_str)

    @property    
    def state(self):
        return {
            "data": self.data.spatial_scale,
            "repr": self.repr.state,
            "structOptim": self.structOptim.state,
            "paramOptim": self.paramOptim.state,
            "iteration": self.iteration
        }        

    def train(self):
        ema_loss_for_log = 0.0
        viewpoint_stack = None
        is_white_background = self.renderer.background_color == [255,255,255]
        for iteration in range(1, self.iterations+1):    
            self.paramOptim.update_lr(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.repr.increment_sh_degree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.data.get_train_pair_list().copy()
            viewpoint_pair = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            render_pkg = self.renderer.render(repr = self.repr, camera = viewpoint_pair.camera)

            # Loss
            gt_image = viewpoint_pair.image.get_resolution_data_from_path(self.data.resolution, self.data.resolution_scales[0])
            loss = self.loss(render_pkg["render"], gt_image)
            loss.backward()
            self.iteration = iteration

            with torch.no_grad():
                
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                self.progress_bar.update(iteration, Loss=ema_loss_for_log)


                # Log and save
                if iteration in self.save_iterations:
                    self.save_scene(iteration)

                # Densification
                self.structOptim.update_optim(iteration, self.repr, self.paramOptim, render_pkg, is_white_background)
                
                # Optimizer step
                self.paramOptim.update_optim()

                # Recorder step
                # self.recorder.update(iteration)

                # Checkpoint saving step
                if iteration in self.ckpt_iterations:
                    pass
                    # self.save_ckpt(iteration)

    def save_scene(self, iteration):
        self.logger.info(f"Saving Gaussians in ITER {iteration}")
        ply_path = self.result_path / f"point_cloud/iteration_{iteration}" / "point_cloud.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        self.repr.save_ply(ply_path)

    






