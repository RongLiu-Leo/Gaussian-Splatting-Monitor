import socket
import json
import struct
import torch
import traceback
from model.base import BaseModule
from utils import BasicCamera

class NetworkGUI(BaseModule):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)
        self.conn = None

    def process_iter(self, renderer, repr, data, iteration, max_iteration):
        
        if self.conn == None:
            self.try_connect()
        while self.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, _, _, keep_alive, _, render_mode_id = self.receive()
                if custom_cam != None:
                    render_mode = self.render_modes[render_mode_id].lower()
                    net_image = renderer.render_img(repr = repr, camera = custom_cam, render_mode = render_mode)
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                self.send(net_image_bytes, data.source_path)
                if do_training and ((iteration < max_iteration) or not keep_alive):
                    break
            except Exception as e:
                raise e
                self.conn = None

    def try_connect(self):
        try:
            self.conn, self.addr = self.listener.accept()
            print(f"\nConnected by {self.addr}")
            self.conn.settimeout(None)
            self.send_render_items(self.render_modes)
        except Exception as inst:
            pass
            # raise inst

    def send_render_items(self, string_list):
        # Serialize the list of strings to JSON
        serialized_data = json.dumps(string_list)
        # Convert the serialized data to bytes
        bytes_data = serialized_data.encode('utf-8')
        # Send the length of the serialized data first
        self.conn.sendall(struct.pack('I', len(bytes_data)))
        # Send the actual serialized data
        self.conn.sendall(bytes_data)

    def read(self):
        messageLength = self.conn.recv(4)
        messageLength = int.from_bytes(messageLength, 'little')
        message = self.conn.recv(messageLength)
        return json.loads(message.decode("utf-8"))

    def send(self, message_bytes, verify):
        if message_bytes != None:
            self.conn.sendall(message_bytes)
        self.conn.sendall(len(verify).to_bytes(4, 'little'))
        self.conn.sendall(bytes(verify, 'ascii'))

    def receive(self):
        message = self.read()

        width = message["resolution_x"]
        height = message["resolution_y"]

        if width != 0 and height != 0:
            try:
                do_training = bool(message["train"])
                fovy = message["fov_y"]
                fovx = message["fov_x"]
                znear = message["z_near"]
                zfar = message["z_far"]
                do_shs_python = bool(message["shs_python"])
                do_rot_scale_python = bool(message["rot_scale_python"])
                keep_alive = bool(message["keep_alive"])
                scaling_modifier = message["scaling_modifier"]
                world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4))
                world_view_transform[:,1] = -world_view_transform[:,1]
                world_view_transform[:,2] = -world_view_transform[:,2]

                custom_cam = BasicCamera(world_view_transform = world_view_transform,
                                         height=height, width=width, 
                                         fov_x=fovx, fov_y=fovy, 
                                         zfar=zfar, znear=znear, 
                                         device="cuda")
                render_mode_id = message["render_mode"]
            except Exception as e:
                print("")
                traceback.print_exc()
                # raise e
            return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier, render_mode_id
        else:
            return None, None, None, None, None, None, None
    
