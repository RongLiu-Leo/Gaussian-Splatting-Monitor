from pathlib import Path
import shutil
from utils import BaseModule

class ColmapProcessor(BaseModule):
    def __init__(self, cfg, logger) -> None:
        super().__init__(cfg, logger)
        self.colmap_command = '"{}"'.format(self.colmap_executable) if len(self.colmap_executable) > 0 else "colmap"
        self.magick_command = '"{}"'.format(self.magick_executable) if len(self.magick_executable) > 0 else "magick"
        self.use_gpu = 1 if not self.no_gpu else 0
    
    def run(self):
        if not self.skip_matching:
            Path(self.source_path, "distorted", "sparse").mkdir(parents=True, exist_ok=True)

            ## Feature extraction
            feat_extracton_cmd = self.colmap_command + " feature_extractor" + \
                    f" --database_path {self.source_path}/distorted/database.db" + \
                    f" --image_path {self.source_path}/input" + \
                    f" --ImageReader.single_camera 1" + \
                    f" --ImageReader.camera_model {self.camera}" + \
                    f" --SiftExtraction.use_gpu {self.use_gpu}"
            self.run_command_with_realtime_output(feat_extracton_cmd)
            
            ## Feature matching
            feat_matching_cmd = self.colmap_command + " exhaustive_matcher" + \
                    f" --database_path {self.source_path}/distorted/database.db" + \
                    f" --SiftMatching.use_gpu {self.use_gpu}"
            self.run_command_with_realtime_output(feat_matching_cmd)

            ### Bundle adjustment
            # The default Mapper tolerance is unnecessarily large,
            # decreasing it speeds up bundle adjustment steps.
            mapper_cmd = self.colmap_command + " mapper" + \
                    f" --database_path {self.source_path}/distorted/database.db" + \
                    f" --image_path {self.source_path}/input" + \
                    f" --output_path {self.source_path}/distorted/sparse" + \
                    f" --Mapper.ba_global_function_tolerance={self.map_ba_global_function_tolerance}"
            self.run_command_with_realtime_output(mapper_cmd)
       
        ### Image undistortion
        ## We need to undistort our images into ideal pinhole intrinsics.
        img_undist_cmd = self.colmap_command + " image_undistorter" + \
                      f" --image_path {self.source_path}/input" + \
                      f" --input_path {self.source_path}/distorted/sparse/0" + \
                      f" --output_path {self.source_path}" + \
                      f" --output_type COLMAP"
        self.run_command_with_realtime_output(img_undist_cmd)

        # Move the undistorted images to the correct location
        sparse_result_path = Path(self.source_path) / "sparse" 
        destination_path = sparse_result_path / "0"
        destination_path.mkdir(exist_ok=True)

        for file in sparse_result_path.iterdir():
            if file.name == '0':
                continue
            source_file = file
            destination_file = destination_path / file.name
            source_file.rename(destination_file)

        if self.resize:
            print("Copying and resizing...")

            # Resize images.
            Path(self.source_path, "images_2").mkdir(parents=True, exist_ok=True)
            Path(self.source_path, "images_4").mkdir(parents=True, exist_ok=True)
            Path(self.source_path, "images_8").mkdir(parents=True, exist_ok=True)
            # Get the list of files in the source directory
            files = [file.name for file in Path(self.source_path, "images").iterdir()]
            # Copy each file from the source directory to the destination directory
            for file in files:
                source_file = Path(self.source_path, "images", file)

                destination_file = Path(self.source_path, "images_2", file)
                shutil.copy2(source_file, destination_file)
                command = self.magick_command + " mogrify -resize 50% " + destination_file
                self.run_command_with_realtime_output(command)

                destination_file = Path(self.source_path, "images_4", file)
                shutil.copy2(source_file, destination_file)
                command = self.magick_command + " mogrify -resize 25% " + destination_file
                self.run_command_with_realtime_output(command)

                destination_file = Path(self.source_path, "images_8", file)
                shutil.copy2(source_file, destination_file)
                command = self.magick_command + " mogrify -resize 12.5% " + destination_file
                self.run_command_with_realtime_output(command)

        self.logger.info("Finish data processing...")
    


    




