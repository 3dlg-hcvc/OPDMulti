
import os
import time
import json
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from multiscan.utils import io, memory_limit, NoDaemonPool
from core import Mask2DCore, CameraParameter
from stream_decode import StreamDecode
from project2d import Project2D
from articulation_in_camera import ArticulationInCamera

log = logging.getLogger(__name__)

import pdb

class Mask2D(Mask2DCore):
    def __init__(self):
        super().__init__()
        self.virtual_camera = False
        self.cpu_num = 0
        self.image_width = 256
        self.image_height = 192

    def use_virtual_camera(self, virtual=True):
        self.virtual_camera = virtual

    def set_cameras(self, cameras: list):
        self.cameras = cameras

    def set_image_resolution(self, width=0, height=0):
        self.image_width = width
        self.image_height = height

    def set_cpu_num(self, cpu_num=0):
        self.cpu_num = cpu_num

    def load_metadata(self, filepath):
        super().set_metadata_path(filepath)
        raw_metadata = io.read_json(filepath)
        self.metadata.color_width = raw_metadata['streams'][0]['resolution'][1]
        self.metadata.color_height = raw_metadata['streams'][0]['resolution'][0]
        self.metadata.depth_width = raw_metadata['streams'][1]['resolution'][1]
        self.metadata.depth_height = raw_metadata['streams'][1]['resolution'][0]
        self.metadata.num_frames = min(
            raw_metadata['streams'][1]['number_of_frames'],
            raw_metadata['streams'][0]['number_of_frames'])
        self.metadata.intrinsics = np.asarray(raw_metadata['streams'][0]['intrinsics'])
        self.metadata.depth_format = raw_metadata.get('depth_unit', 'm')

    def load_camera_from_trajectory(self, frame_indices=[]):
        if not len(frame_indices):
            frame_indices = list(range(0, self.metadata.num_frames, 1))
        start_time = time.time()
        self.cameras = []
        with open(self.cameras_path, 'r') as pose_file:
            lines = pose_file.readlines()
            for frame_idx in frame_indices:
                line = lines[frame_idx]
                cam_info = json.loads(line)
                C = cam_info.get('transform', None)
                # ARKit pose (+x along long axis of device toward home button, +y upwards, +z away from device)
                assert C != None
                C = np.asarray(C)
                C = C.reshape(4, 4).transpose()

                C = np.matmul(C, np.diag([1, 1, 1, 1]))  # open3d camera pose (flip y and z)
                C = C / C[3][3]

                extrinsics = np.linalg.inv(C)

                K = cam_info.get('intrinsics', None)
                assert K != None

                K = np.asarray(K).reshape(3, 3).transpose()

                scale_d2c_x = float(self.metadata.depth_width) / self.metadata.color_width
                scale_d2c_y = float(self.metadata.depth_height) / self.metadata.color_height
                
                scale_w = self.image_width / self.metadata.depth_width
                scale_h = self.image_height / self.metadata.depth_height
                scale = np.array([scale_d2c_x * scale_w, scale_d2c_y * scale_h, 1.0])

                K_depth = np.matmul(np.diag(scale), K)
                intrinsics = K_depth

                cam_param = CameraParameter(intrinsics, extrinsics, name=str(frame_idx))
                self.cameras.append(cam_param)

        log.debug("--- %s seconds read camera extrinsics and intrinsics ---" % (time.time() - start_time))
        return self.cameras

    def export_rgb_images(self, output_dir, frame_indices=[]):
        super().set_output_rgb_dir(output_dir)
        if not self.virtual_camera:
            stream_decode = StreamDecode(cpu_num=self.cpu_num)
            stream_decode.set_metadata(self.metadata)
            stream_decode.set_rgb_path(self.rgb_path)
            stream_decode.set_image_resolution(self.image_width, self.image_height)
            stream_decode.export_rgb_images(output_dir, frame_indices, image_format='.png')
        # else:
            # self.load_camera_from_trajectory(frame_indices)
            # alignment = io.read_json(self.align_path)
            # transform = np.reshape(alignment['coordinate_transform'], (4, 4), order='F')

            # project_2d = Project2D()
            # project_2d.set_metadata(self.metadata)
            # project_2d.load_ply(self.ply_path, transform)
            # project_2d.load_obj(self.textured_mesh_path, transform)
            # project_2d.set_cameras(self.cameras)
            # project_2d.set_window_size(self.image_width, self.image_height)
            # project_2d._initialize_render_cameras()
            # project_2d.export_rgb_images(output_dir, ext='.png')


    def export_depth_images(self, output_dir, frame_indices=[]):
        super().set_output_depth_dir(output_dir)
        if not self.virtual_camera:
            stream_decode = StreamDecode(cpu_num=self.cpu_num)
            stream_decode.set_metadata(self.metadata)
            stream_decode.set_depth_path(self.depth_path)
            stream_decode.set_image_resolution(self.image_width, self.image_height)
            stream_decode.export_depth_images(output_dir, frame_indices, image_format='.png')

    def export_mask_images(self, output_dir, frame_indices=[]):
        super().set_output_mask_dir(output_dir)
        if not self.virtual_camera:
            self.load_camera_from_trajectory(frame_indices)
            alignment = io.read_json(self.align_path)
            transform = np.reshape(alignment['coordinate_transform'], (4, 4), order='F')

            project_2d = Project2D()
            project_2d.set_metadata(self.metadata)
            project_2d.load_ply(self.ply_path, transform)
            # project_2d.load_obj(self.textured_mesh_path, transform)
            project_2d.set_cameras(self.cameras)
            project_2d.set_window_size(self.image_width, self.image_height)
            project_2d._initialize_render_cameras()
            project_2d.export_mask_images(output_dir, ext='.png')
            # pdb.set_trace()
            # project_2d.export_rgb_images(output_dir, ext='.png')

    def export_annotations(self, output_dir, frame_indices=[]):
        super().set_output_articulation_dir(output_dir)
        if not self.virtual_camera:
            self.load_camera_from_trajectory(frame_indices)
            alignment = io.read_json(self.align_path)
            transform = np.reshape(alignment['coordinate_transform'], (4, 4), order='F')

            art_in_camera = ArticulationInCamera()
            art_in_camera.set_metadata(self.metadata)
            art_in_camera.set_cameras(self.cameras)
            art_in_camera.set_additional_transformation(transform)
            art_in_camera.load_annotations(self.annotations_path)
            art_in_camera.export_annotations(output_dir)

def process_scan(cfg, scan_id):
    sucess = True
    start = time.time()
    num_exported, num_total = 0, 0
    try:
        log.info(f'start processing scan {scan_id}')
        mask_2d = Mask2D()

        # setup paths
        mask_2d.set_rgb_path(cfg.rgb_path.format(scanId=scan_id))
        mask_2d.set_depth_path(cfg.depth_path.format(scanId=scan_id))
        mask_2d.set_confidence_path(cfg.confidence_path.format(scanId=scan_id))
        mask_2d.set_ply_path(cfg.ply_path.format(scanId=scan_id))
        mask_2d.set_align_path(cfg.align_path.format(scanId=scan_id))
        mask_2d.set_cameras_path(cfg.cameras_path.format(scanId=scan_id))
        mask_2d.set_textured_mesh_path(cfg.textured_mesh_path.format(scanId=scan_id))
        mask_2d.set_annotations_path(cfg.annotations_path.format(scanId=scan_id))

        mask_2d.use_virtual_camera(virtual=cfg.virtual_camera)
        mask_2d.set_cpu_num(cfg.cpu_num)
        mask_2d.load_metadata(cfg.metadata_path.format(scanId=scan_id))
        mask_2d.set_image_resolution(cfg.image_width, cfg.image_height)

        # output
        # pdb.set_trace()
        frame_indices = np.arange(0, mask_2d.metadata.num_frames, cfg.step)
        rgb_image_paths = mask_2d.export_rgb_images(cfg.output_rgb_dir.format(scanId=scan_id), frame_indices)
        depth_image_paths = mask_2d.export_depth_images(cfg.output_depth_dir.format(scanId=scan_id), frame_indices)
        mask_image_paths = mask_2d.export_mask_images(cfg.output_mask_dir.format(scanId=scan_id), frame_indices)
        # TODO: ADd obb to articulation annotations
        articulation_paths = mask_2d.export_annotations(cfg.output_annotation_dir.format(scanId=scan_id), frame_indices)
    except Exception as e:
        sucess = False
        log.error(e)
        raise e
    duration = time.time() - start
    
    status = 'sucess' if sucess else 'faild'

    df_row = pd.DataFrame(
            [[scan_id, status, duration, num_exported, num_total]],
            columns=['scanId', 'status', 'duration', 'framesExported', 'framesTotal'])
    return df_row

@memory_limit(percentage=0.9)
@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg : DictConfig):
    scan_ids = io.get_folder_list(cfg.input_dir, join_path=False)
    # scan_ids = ["scene_00013_00"]
    # pdb.set_trace()
    if 'anonymous_data' in scan_ids:
        scan_ids.remove('anonymous_data')
    # pdb.set_trace()
    if cfg.debug:
        scan_ids = scan_ids[:4]
    io.ensure_dir_exists(cfg.output_dir)

    df_list = []
    if cfg.cpu_num > 1:
        pool = NoDaemonPool(processes=cfg.cpu_num)
        jobs = [pool.apply_async(process_scan, args=(cfg, scan_id,)) for scan_id in scan_ids]
        pool.close()
        pool.join()
        df_row_list = [job.get() for job in jobs]
        df_list += df_row_list
    else:
        for scan_id in tqdm(scan_ids):
            df_row = process_scan(cfg, scan_id)
            df_list.append(df_row)
            break
    df = pd.concat(df_list)
    df.to_csv(os.path.join(cfg.output_dir, 'processed.csv'))

if __name__ == "__main__":
    main()