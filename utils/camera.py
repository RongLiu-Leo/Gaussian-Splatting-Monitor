import torch
from torch import nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils import fov2focal, getWorld2View, getProjectionMatrix

class BasicCamera():
    def __init__(self, fov_x, fov_y, height, width, 
                 R = None, T = None, world_view_transform = None,
                 z_far = 100.0, z_near = 0.01, uid = 0, device = 'cuda',
                 **kwargs):
        super(BasicCamera, self).__init__()


        
        self.fov_x, self.fov_y = fov_x, fov_y
        self.width, self.height = width, height
        self.z_far = z_far
        self.z_near = z_near
        self.uid = uid
        self.device = device
        self.trans=np.array([0.0, 0.0, 0.0])
        self.scale=1.0
        for key, value in kwargs.items():
            setattr(self, key, value)
        if world_view_transform is not None:
            self.world_view_transform = world_view_transform
            # TODO: check if this is correct
            # self.R = self.world_view_transform[:3, :3].cpu().numpy()
            # self.T = self.world_view_transform[:3, 3].cpu().numpy()
        elif R is not None and T is not None:
            self.R = R
            self.T = T
            self.world_view_transform = torch.tensor(getWorld2View(R, T, self.trans, self.scale)).to(self.device)
        else:
            raise ValueError("Either R and T or world_view_transform must be provided.")
    @property
    def projection_matrix(self):
        return getProjectionMatrix(znear=self.z_near, zfar=self.z_far, fovX=self.fov_x, fovY=self.fov_y).transpose(0,1).to(self.device)
    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(self.device)
    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3].to(self.device)

class BasicImage:
    def __init__(self, data=None, device = 'cuda', path = None, name = None, gt_alpha_mask = None, keep_data = False, **kwargs):
        # data is a [channels, height, width] tensor
        self.device = device
        self.data = self.format_data(data, gt_alpha_mask)
        self.gt_alpha_mask = gt_alpha_mask
        self.channels, self.height, self.width = self.data.shape
        self.path, self.name = path,name
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not keep_data:
            self.data = None
        self.resolution_data_dict = {}
        
    @staticmethod
    def format_data(data, gt_alpha_mask):
        if data is None:
            return None
        # Convert data to a PyTorch tensor
        if isinstance(data, Image.Image):
            np_image = np.array(data)
            if np_image.shape[2] == 4 and gt_alpha_mask is None:
                gt_alpha_mask = np_image[:, :, 3]
                gt_alpha_mask = torch.from_numpy(gt_alpha_mask) / 255.0
                np_image = np_image[:, :, :3]
            image = torch.from_numpy(np_image) / 255.0
            data = image.permute(2, 0, 1)
        elif isinstance(data, np.ndarray):
            if data.shape[2] == 4 and gt_alpha_mask is None:
                gt_alpha_mask = np_image[:, :, 3]
                gt_alpha_mask = torch.from_numpy(gt_alpha_mask) / 255.0
                np_image = np_image[:, :, :3]
            data = torch.from_numpy(data) / 255.0
        elif isinstance(data, torch.Tensor):
            data = data.clone().detach()
            if data.max() > 1.0:
                data = data.float() / 255
        else:
            print(f"Image data should be in [Image.Image, np.ndarray, torch.Tensor], but get {type(data)}")
        
        # Multiply gt_alpha_mask
        if gt_alpha_mask is not None:
            data *= gt_alpha_mask

        return data
    
    def to_device(self, device):
        self.device = device
        self.data = self.data.to(device)

    def set(self,**kwargs):
        if 'data' in kwargs:
            self.data = self.format_data(kwargs['data'], self.gt_alpha_mask)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_data_from_path(self):
        data = Image.open(self.path)
        data = self.format_data(data, self.gt_alpha_mask)
        data = data.to(self.device)
        return data

    def get_resolution(self, resolution_input,resolution_scale):
        orig_w, orig_h = self.width, self.height
        # resolution in (height, width)
        if resolution_input in [1, 2, 4, 8]:
            resolution = round(orig_h/(resolution_scale * resolution_input)), round(orig_w/(resolution_scale * resolution_input))
        else:  # should be a type that converts to float
            if resolution_input == -1:
                # if orig_w > 1600:
                #     global_down = orig_w / 1600
                # else:
                global_down = 1
            else:
                global_down = orig_w / resolution_input

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_h / scale), int(orig_w / scale))
        return resolution

    def get_resolution_data_from_path(self, resolution_input, resolution_scale):
        resolution = self.get_resolution(resolution_input, resolution_scale)
        if self.resolution_data_dict.get(resolution) is None:
            resize_transform = transforms.Resize(resolution)
            data = Image.open(self.path)
            data = self.format_data(data, self.gt_alpha_mask)
            resized_image_tensor = resize_transform(data.unsqueeze(0)).squeeze(0)
            data = resized_image_tensor.to(self.device)
            self.resolution_data_dict[resolution] = data
        data = self.resolution_data_dict.get(resolution)
        return data

class CameraImagePair:
    def __init__(self, cam: BasicCamera, img: BasicImage, uid: int, **kwargs):
        self.camera = cam
        self.image = img
        self.uid = uid
        for key, value in kwargs.items():
            setattr(self, key, value)
    @property
    def json(self):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.camera.R.transpose()
        Rt[:3, 3] = self.camera.T
        Rt[3, 3] = 1.0

        W2C = np.linalg.inv(Rt)
        pos = W2C[:3, 3]
        rot = W2C[:3, :3]
        serializable_array_2d = [x.tolist() for x in rot]
        return  {
            'id' : self.uid,
            'img_name' : self.image.name,
            'width' : self.camera.width,
            'height' : self.camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : fov2focal(self.camera.fov_y, self.camera.height),
            'fx' : fov2focal(self.camera.fov_x, self.camera.width)
        }
    def set(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)



