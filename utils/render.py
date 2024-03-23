from utils import depth_to_normal,gradient_map,colormap

def render_net_image(render_pkg, render_mode, camera):
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