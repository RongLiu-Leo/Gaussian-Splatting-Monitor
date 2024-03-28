from utils import *
from dataloader.colmap_helper import load_colmap_folder, get_spatial_scale
import random

class ColmapData(BaseModule):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.point_cloud, pair_list, self.ply_path = load_colmap_folder(self.source_path)

        # Define train and test dataset
        if self.eval > 0:
            self._train_pair_list = [c for idx, c in enumerate(pair_list) if idx % self.eval != 0]
            self._test_pair_list = [c for idx, c in enumerate(pair_list) if idx % self.eval == 0]
        else:
            self._train_pair_list = pair_list
            self._test_pair_list = []

        self.spatial_scale = get_spatial_scale(self._train_pair_list)["radius"]

        # Shuffle the dataset
        if self.shuffle:
            random.shuffle(self._train_pair_list)  # Multi-res consistent random shuffling
            random.shuffle(self._test_pair_list)  # Multi-res consistent random shuffling
    
    def get_train_pair_list(self):
        return self._train_pair_list

    def get_test_pair_list(self):
        return self._test_pair_list
    
        