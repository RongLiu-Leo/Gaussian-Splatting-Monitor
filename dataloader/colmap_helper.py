import collections
import numpy as np
from pathlib import Path
import struct
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from pathlib import Path
from utils import *
from tqdm import tqdm

GCameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])

CAMERA_MODELS = {
    GCameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    GCameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    GCameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    GCameraModel(model_id=3, model_name="RADIAL", num_params=5),
    GCameraModel(model_id=4, model_name="OPENCV", num_params=8),
    GCameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    GCameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    GCameraModel(model_id=7, model_name="FOV", num_params=5),
    GCameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    GCameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    GCameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

GCAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])



class ColmapCamera:
    def __init__(self, uid: int,
                       model: str, 
                       width: int, 
                       height: int, 
                       params: np.array):
        self.uid = uid
        self.model = model
        self.params = params
        self.width = width
        self.height = height
    
class ColmapImage:
    def __init__(self,uid: int,
                    qvec: np.array,
                    tvec: np.array,
                    xys: np.array,
                    point3D_ids: np.array,
                    camera_id: int,
                    name: str):
        self.uid = uid
        self.camera_id = camera_id
        self.name = name
        self.qvec = qvec
        self.tvec = tvec
        self.point3D_ids = point3D_ids
        self.xys = xys

    @property
    def R(self):
        return np.transpose(qvec2rotmat(self.qvec))
    @property
    def T(self):
        return np.array(self.tvec)



def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ColmapImage(
                uid=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = GCAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = GCAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = ColmapCamera(uid=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = ColmapCamera(uid=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = ColmapImage(
                    uid=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def get_colmap_camera_image_pair_list(cam_extrinsics, cam_intrinsics, images_folder):
    cam_img_pair_list = []
    for idx, key in tqdm(enumerate(cam_extrinsics.keys()), total=len(cam_extrinsics)):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            fov_y = focal2fov(focal_length_x, intr.height)
            fov_x = focal2fov(focal_length_x, intr.width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            fov_y = focal2fov(focal_length_y, intr.height)
            fov_x = focal2fov(focal_length_x, intr.width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        camera = BasicCamera(R=extr.R, T=extr.T, fov_y=fov_y, fov_x=fov_x, width=intr.width, height=intr.height,
                            uid=intr.uid)
        image_path = Path(images_folder) / extr.name
        image = BasicImage(data = Image.open(str(image_path)), path=image_path, name=extr.name)
        cam_img_pair = CameraImagePair(cam=camera, img = image, uid=idx)
        cam_img_pair_list.append(cam_img_pair)

    return cam_img_pair_list

def get_spatial_scale(cam_image_pair_list):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam_image_pair in cam_image_pair_list:
        cam = cam_image_pair.camera
        W2C = getWorld2View(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def load_colmap_folder(colmap_folder):
    try:
        cameras_extrinsic_file = Path(colmap_folder) / "sparse" / "0" / "images.bin"
        cameras_intrinsic_file = Path(colmap_folder) / "sparse" / "0" / "cameras.bin"
        cam_extrinsics = read_extrinsics_binary(str(cameras_extrinsic_file))
        cam_intrinsics = read_intrinsics_binary(str(cameras_intrinsic_file))
    except:
        cameras_extrinsic_file = Path(colmap_folder) / "sparse" / "0" / "images.txt"
        cameras_intrinsic_file = Path(colmap_folder) / "sparse" / "0" / "cameras.txt"
        cam_extrinsics = read_extrinsics_text(str(cameras_extrinsic_file))
        cam_intrinsics = read_intrinsics_text(str(cameras_intrinsic_file))
    images_folder = Path(colmap_folder) / "images"
    pair_list_unsorted = get_colmap_camera_image_pair_list(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=images_folder)
    pair_list = sorted(pair_list_unsorted.copy(), key = lambda x : x.image.name)


    ply_path = str(Path(colmap_folder) / "sparse" / "0" / "points3D.ply")
    bin_path = str(Path(colmap_folder) / "sparse" / "0" / "points3D.bin")
    txt_path = str(Path(colmap_folder) / "sparse" / "0" / "points3D.txt")
    if not Path(ply_path).exists():
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    return pcd, pair_list, ply_path
