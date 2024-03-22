import math
import torch
from utils import eval_sh
from model.base import BaseModule
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class DiffRasterizerRenderer(BaseModule):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self._background_color = cfg.get("background_color", [0, 0, 0])
        
    @property
    def background_color(self):
        if self._background_color == [-1,-1,-1]: # random background
            return torch.rand((3), device="cuda")
        else:
            return torch.tensor(self._background_color, dtype=torch.float32, device="cuda")

    def render(self, repr, camera):
        """
        Render the scene. 
        Background tensor (bg_color) must be on GPU!
        """
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(repr.xyz, dtype=repr.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(camera.fov_x * 0.5)
        tanfovy = math.tan(camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=repr.sh_degree,
            campos=camera.camera_center,
            prefiltered=self.prefiltered,
            debug=self.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = repr.xyz
        means2D = screenspace_points
        opacity = repr.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python:
            cov3D_precomp = repr.covariance(self.scaling_modifier)
        else:
            scales = repr.scaling
            rotations = repr.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.override_color == [-1,-1,-1]:
            if self.convert_SHs_python:
                shs_view = repr.features.transpose(1, 2).view(-1, 3, (repr.max_sh_degree+1)**2)
                dir_pp = (repr.xyz - camera.camera_center.repeat(repr.features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(repr.sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = repr.features
        else:
            colors_precomp = self.override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, rendered_depth, rendered_median_depth, rendered_alpha, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "mean_depth": rendered_depth,
            "median_depth": rendered_median_depth,
            "alpha": rendered_alpha}