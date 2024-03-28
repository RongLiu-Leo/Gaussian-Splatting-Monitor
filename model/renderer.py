import math
import torch
from utils import eval_sh,depth_to_normal,gradient_map,colormap
from utils.base import BaseModule
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

    def render_pkg(self, repr, camera):
        """
        Render the scene. 
        Background tensor (bg_color) must be on GPU!
        """
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(repr.xyz, dtype=repr.xyz.dtype, requires_grad=True).cuda() + 0
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
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = repr.xyz
        means2D = screenspace_points
        opacity = repr.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = repr.covariance(self.scaling_modifier)


        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        shs_view = repr.features.transpose(1, 2).view(-1, 3, (repr.max_sh_degree+1)**2)
        dir_pp = (repr.xyz - camera.camera_center.repeat(repr.features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(repr.sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

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

    def render_img(self, repr, camera, render_mode):
        render_pkg = self.render_pkg(repr, camera)
        if render_mode == 'alpha':
            net_image = render_pkg["alpha"]
            net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
        elif render_mode == 'depth':
            net_image = render_pkg["mean_depth"]
            net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
        elif render_mode == 'normal':
            net_image = depth_to_normal(render_pkg["mean_depth"], camera).permute(2,0,1)
            net_image = (net_image+1)/2
        elif render_mode == 'edge':
            net_image = gradient_map(render_pkg["render"])
        elif render_mode == 'curvature':
            net_image = gradient_map(depth_to_normal(render_pkg["mean_depth"], camera).permute(2,0,1))
        else:
            net_image = render_pkg["render"]
        if net_image.shape[0]==1:
            net_image = colormap(net_image)
        return net_image
        