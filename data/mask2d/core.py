from dataclasses import dataclass
import numpy as np
from multiscan.utils import io
import pdb

@dataclass
class StreamMetaData:
    color_height: int = 0
    color_width: int = 0
    depth_height: int = 0
    depth_width: int = 0
    num_frames: int = 0
    intrinsics: np.ndarray = np.eye(3)
    depth_format: str = 'm'

@dataclass
class CameraParameter:
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    name: str = '0'

class Mask2DCore:
    def __init__(self):
        self.metadata = StreamMetaData()
        self.cameras = []

        self.ply_path = None
        self.rgb_path = None
        self.depth_path = None
        self.confidence_path = None
        self.metadata_path = None
        self.cameras_path = None
        self.annotations_path = None
        self.textured_mesh_path = None

        self.output_rgb_dir = None
        self.output_depth_dir = None
        self.output_mask_dir = None
        self.output_articulation_dir = None

    def _load(self, filepath, ext=''):
        if filepath and io.file_exist(filepath, ext=ext):
            return True
        else:
            return False
    
    def _export(self, output_dir):
        io.ensure_dir_exists(output_dir)

    def set_metadata(self, metadta: StreamMetaData):
        self.metadata = metadta

    def set_ply_path(self, filepath, ext='.ply'):
        if self._load(filepath, ext):
            self.ply_path = filepath
    
    def set_align_path(self, filepath, ext='.json'):
        if self._load(filepath, ext):
            self.align_path = filepath

    def set_rgb_path(self, filepath, ext='.mp4'):
        if self._load(filepath, ext):
            self.rgb_path = filepath

    def set_depth_path(self, filepath, ext='.zlib'):
        if self._load(filepath, ext):
            self.depth_path = filepath

    def set_confidence_path(self, filepath, ext='.zlib'):
        if self._load(filepath, ext):
            self.confidence_path = filepath

    def set_metadata_path(self, filepath, ext='.json'):
        if self._load(filepath, ext):
            self.metadata_path = filepath

    def set_cameras_path(self, filepath, ext='.jsonl'):
        if self._load(filepath, ext):
            self.cameras_path = filepath

    def set_annotations_path(self, filepath, ext='.json'):
        if self._load(filepath, ext):
            self.annotations_path = filepath

    def set_textured_mesh_path(self, filepath, ext='.obj'):
        if self._load(filepath, ext):
            self.textured_mesh_path = filepath

    def set_output_rgb_dir(self, folderpath):
        self._export(folderpath)
        self.output_rgb_dir = folderpath

    def set_output_depth_dir(self, folderpath):
        self._export(folderpath)
        self.output_depth_dir = folderpath

    def set_output_mask_dir(self, folderpath):
        self._export(folderpath)
        self.output_mask_dir = folderpath

    def set_output_articulation_dir(self, folderpath):
        self._export(folderpath)
        self.output_articulation_dir = folderpath
