import torch
import numpy as np
import torch.nn as nn
from model.base import BaseModule


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class AdamWithcustomlrParamOptim(BaseModule):
    def __init__(self, cfg, logger, spatial_lr_scale, repr):
        super().__init__(cfg, logger)
        self.optimizer = torch.optim.Adam(self.param_lr_groups(repr), lr=0.0, eps=1e-15)

        self.xyz_lr_schedule = get_expon_lr_func(lr_init=self.position_lr_init*spatial_lr_scale,
                                                lr_final=self.position_lr_final*spatial_lr_scale,
                                                lr_delay_mult=self.position_lr_delay_mult,
                                                max_steps=self.position_lr_max_steps)

    @property
    def state(self):
        return self.optimizer.state_dict()
    
    def load(self, state):
        self.optimizer.load_state_dict(state)
        
    def param_lr_groups(self, repr):
        param_groups = [
            {'params': [repr._xyz], 'lr': self.position_lr_init * repr.spatial_lr_scale, "name": "xyz"},
            {'params': [repr._features_dc], 'lr': self.feature_lr, "name": "f_dc"},
            {'params': [repr._features_rest], 'lr': self.feature_lr / 20.0, "name": "f_rest"},
            {'params': [repr._opacity], 'lr': self.opacity_lr, "name": "opacity"},
            {'params': [repr._scaling], 'lr': self.scaling_lr, "name": "scaling"},
            {'params': [repr._rotation], 'lr': self.rotation_lr, "name": "rotation"}
        ]
        return param_groups
    
    def update_lr(self,iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_lr_schedule(iteration)
                param_group['lr'] = lr
                return lr
            
    def update_optim(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

    def prune_optim(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def replace_tensor(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def print_state(self):
        state_dict = self.optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            name = param_group['name']
            idx = param_group['params'][0]
            exp_avg = state_dict['state'][idx]['exp_avg']
            exp_avg_sq = state_dict['state'][idx]['exp_avg_sq']
            print(f"{name}: exp_avg-{exp_avg.shape}, exp_avg_sq-{exp_avg_sq.shape}")