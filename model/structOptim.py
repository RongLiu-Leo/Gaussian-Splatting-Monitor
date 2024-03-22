import torch
from model.base import BaseModule
from utils import build_rotation, inverse_sigmoid

class SplitWithCloneWithPrune(BaseModule):
    def __init__(self, cfg, logger, spatial_lr_scale, num_init_points, state = None):
        super().__init__(cfg, logger)
        if state:
            (self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom )  = state
        else:
            self.spatial_lr_scale = spatial_lr_scale
            self.reset_stats(num_init_points)  

    @property
    def state(self):
        return (
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom  
        )
            
    def update_optim(self, iteration, repr, paramOptim, render_pkg, is_white_background):
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if iteration < self.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter,:2], dim=-1, keepdim=True)
            self.denom[visibility_filter] += 1

            if iteration > self.densify_from_iter and iteration % self.densification_interval == 0:
                self.densify_and_prune(iteration, repr, paramOptim)
            if iteration % self.opacity_reset_interval == 0 or (is_white_background and iteration == self.densify_from_iter):
                self.reset_model_opacity(repr, paramOptim)
    
    def should_start_limit_size(self,iteration):
        return iteration > self.opacity_reset_interval

    def densify_and_prune(self, iteration, repr, paramOptim):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(repr, paramOptim, grads)
        self.densify_and_split(repr, paramOptim, grads)

        prune_mask = (repr.opacity < self.min_opacity).squeeze()
        if self.should_start_limit_size(iteration):
            big_points_vs = self.max_radii2D > self.size_threshold
            big_points_ws = repr.scaling.max(dim=1).values > 0.1 * self.spatial_lr_scale
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, repr, paramOptim)
        torch.cuda.empty_cache()

    def densify_and_clone(self, repr, paramOptim, grads):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= self.densify_grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(repr.scaling, dim=1).values <= self.percent_dense*self.spatial_lr_scale)
        
        new_tensors_dict = {
            "xyz": repr._xyz[selected_pts_mask],
            "f_dc": repr._features_dc[selected_pts_mask],
            "f_rest": repr._features_rest[selected_pts_mask],
            "opacity": repr._opacity[selected_pts_mask],
            "scaling" : repr._scaling[selected_pts_mask],
            "rotation" : repr._rotation[selected_pts_mask]
        }
        
        self.densification_postfix(repr, paramOptim, new_tensors_dict)

    def densify_and_split(self, repr, paramOptim, grads):
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((repr.xyz.shape[0]), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= self.densify_grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(repr.scaling, dim=1).values > self.percent_dense*self.spatial_lr_scale)

        stds = repr.scaling[selected_pts_mask].repeat(self.num_split,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(repr._rotation[selected_pts_mask]).repeat(self.num_split,1,1)
        
        new_tensors_dict = {
            "xyz": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + repr.xyz[selected_pts_mask].repeat(self.num_split, 1),
            "f_dc": repr._features_dc[selected_pts_mask].repeat(self.num_split,1,1),
            "f_rest": repr._features_rest[selected_pts_mask].repeat(self.num_split,1,1),
            "opacity": repr._opacity[selected_pts_mask].repeat(self.num_split,1),
            "scaling" : repr.scaling_inverse_activation(repr.scaling[selected_pts_mask].repeat(self.num_split,1) / (0.8*self.num_split)),
            "rotation" : repr._rotation[selected_pts_mask].repeat(self.num_split,1)
        }

        self.densification_postfix(repr, paramOptim, new_tensors_dict)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(self.num_split * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, repr, paramOptim)

    def prune_points(self, mask, repr, paramOptim):
        valid_points_mask = ~mask
        optimizable_tensors = paramOptim.prune_optim(valid_points_mask)
        repr.update_params(optimizable_tensors)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, repr, paramOptim, new_tensors_dict):
        optimizable_tensors = paramOptim.cat_tensors(new_tensors_dict)
        repr.update_params(optimizable_tensors)
        self.reset_stats(repr.xyz.shape[0])

    def reset_model_opacity(self, repr, paramOptim):
        opacities_new = inverse_sigmoid(torch.min(repr.opacity, torch.ones_like(repr.opacity)*0.01))
        optimizable_tensors = paramOptim.replace_tensor(opacities_new, "opacity")
        repr._opacity = optimizable_tensors["opacity"]

    def reset_stats(self, num_points):
        self.xyz_gradient_accum = torch.zeros((num_points, 1)).cuda()
        self.denom = torch.zeros((num_points, 1)).cuda()
        self.max_radii2D = torch.zeros((num_points)).cuda()

