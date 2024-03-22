import torch
from random import randint
from model.base import BaseModule
from pathlib import Path


class BaseTrainer(BaseModule):
    def __init__(self, cfg, logger, data, repr, loss, renderer, paramOptim, structOptim):
        super().__init__(self, cfg, logger)
        self.data = data
        self.repr = repr
        self.loss = loss
        self.paramOptim = paramOptim
        self.renderer = renderer
        self.structOptim = structOptim
        
        

    @property    
    def state(self):
        return {
            "data": self.data.spatial_scale,
            "representation": self.representation.state,
            "structOptim": self.structOptim.state,
            "paramOptim": self.paramOptim.state,
            "iteration": self.iteration
        }        

    def restore_components(self, system_path, iteration):
        ckpt_path = Path(system_path) / f"{iteration}.pth"
        try:
            ckpt_dict = torch.load(ckpt_path)
            spatial_lr_scale = ckpt_dict["data"]
            
            pcd_path = Path(self.view_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
            self.representation.load_ply(str(pcd_path))
            self.representation.restore(state = ckpt_dict["representation"], spatial_lr_scale = spatial_lr_scale)
            
            self.structOptim.restore(state = ckpt_dict["structOptim"], spatial_lr_scale = spatial_lr_scale)
            
            param_lr_group = self.representation.create_param_lr_groups(self.paramOptim.cfg)
            self.paramOptim.restore(state = ckpt_dict["paramOptim"], spatial_lr_scale = spatial_lr_scale, param_lr_group = param_lr_group, max_iter = self.cfg.iterations)
            
            self.first_iteration = ckpt_dict["iteration"] + 1
            # init progress bar
            self.progress_bar = ProgressBar(first_iter=self.first_iteration, total_iters=self.cfg.iterations)
            
        except Exception as e:
            self.logger.warning(f"Cannot load {ckpt_path}! Error: {e} Train from scratch")
            self.setup_components()

    def train(self):
        ema_loss_for_log = 0.0
        viewpoint_stack = None
        is_white_background = self.renderer.background_color == [255,255,255]
        for iteration in range(self.first_iteration, self.first_iteration + self.cfg.iterations + 1):    
            self.paramOptim.update_lr(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.representation.increment_sh_degree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.data.get_train_pair_list().copy()
            viewpoint_pair = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            render_pkg = self.renderer.render(representation = self.representation, camera = viewpoint_pair.camera)

            # Loss
            gt_image = viewpoint_pair.image.get_resolution_data_from_path(self.data.cfg.resolution, self.data.cfg.resolution_scales[0])
            loss = self.loss(render_pkg["render"], gt_image)
            loss.backward()
            self.iteration = iteration

            with torch.no_grad():
                
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                self.progress_bar.update(iteration, ema_loss_for_log=ema_loss_for_log)
                self.recorder.snapshot("ema_loss_for_log", ema_loss_for_log)
                self.recorder.snapshot("loss", loss.clone().detach().cpu().item())
                # if self.recorder.should_process(iteration, name="image"):
                #     viewpoint_stack_all = self.data.get_train_pair_list().copy()
                #     for viewpoint_pair_this in viewpoint_stack_all:
                #         render_pkg1 = self.renderer.render(representation = self.representation, camera = viewpoint_pair_this.camera)
                #         render_pkg2 = self.renderer2.render(representation = self.representation, camera = viewpoint_pair_this.camera)
                #         depth_map_tensor = render_pkg1["depth"].clone().cpu()
                #         depth_normalized = (depth_map_tensor - depth_map_tensor.min()) / (depth_map_tensor.max() - depth_map_tensor.min())
                #         depth_scaled = (depth_normalized * 255).type(torch.uint8)
                #         self.recorder.snapshot("depth-mean", {f"{iteration}-{viewpoint_pair_this.image.name}":depth_scaled})
                #         depth_map_tensor = render_pkg2["depth"].clone().cpu()
                #         depth_normalized = (depth_map_tensor - depth_map_tensor.min()) / (depth_map_tensor.max() - depth_map_tensor.min())
                #         depth_scaled = (depth_normalized * 255).type(torch.uint8)
                #         self.recorder.snapshot("depth-median", {f"{iteration}-{viewpoint_pair_this.image.name}":depth_scaled})   
                #     self.recorder.snapshot("image", {f"{iteration}-{viewpoint_pair_this.image.name}":render_pkg2["render"].clone().cpu()})

                # Log and save
                if iteration in self.cfg.save_iterations:
                    self.save_scene(iteration)

                # Densification
                self.structOptim.update_optim(iteration, self.representation, self.paramOptim, render_pkg, is_white_background)
                
                # Optimizer step
                self.paramOptim.update_optim(iteration)

                # Recorder step
                self.recorder.update(iteration)

                # Checkpoint saving step
                if iteration in self.cfg.ckpt_iterations:
                    self.save_ckpt(iteration)

        

    






