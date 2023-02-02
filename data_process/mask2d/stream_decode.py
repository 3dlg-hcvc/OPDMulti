import os
import argparse
import pdb
import zlib
import copy
import logging
import numpy as np
import utils
from time import time
from enum import Enum
from PIL import Image
from collections import deque
from types import SimpleNamespace
from multiprocessing import Pool, cpu_count
from decord import VideoReader
from decord import cpu as VideoCPU
from core import Mask2DCore

from multiscan.utils import io

log = logging.getLogger(__name__)

class ImageCategory(Enum):
    COLOR = 'color'
    DEPTH = 'depth'
    CONFIDENCE = 'confidence'

class StreamDecode(Mask2DCore):
    def __init__(self, cpu_num=-1):
        super().__init__()

        self.image_cat = ImageCategory.COLOR
        self.frames_batch = 100
        self.colormap = False

        self.cpu_num = cpu_count() if cpu_num <= 0 else cpu_num
        self.cpu_num = min(cpu_count(), self.cpu_num)

        self.confidence_level = 0
        self.delta_threshold = 0
        
        self.step = 1

        self.image_width = 0
        self.image_height = 0

    def set_image_resolution(self, width=0, height=0):
        self.image_width = width
        self.image_height = height

    @staticmethod
    def export_frame(frame_name, frame, output_dir, image_format='.png', width=0, height=0,
                     image_type=ImageCategory.COLOR, scale=1, colormap=False):
        frame = frame * scale
        if image_type == ImageCategory.COLOR:
            img = Image.fromarray(frame.astype(np.uint8))
        elif image_type == ImageCategory.CONFIDENCE:
            if colormap:
                img_arr = utils.image_colormap(frame, colormap='tab10')
                img = Image.fromarray(img_arr.astype(np.uint8))
            else:
                img = Image.fromarray(frame.astype(np.uint8))
        elif ImageCategory.DEPTH:
            if colormap:
                img_arr = utils.image_colormap(frame, colormap='jet', mask=(frame != 0), clamp=3000)
                img = Image.fromarray(img_arr.astype(np.uint8))
            else:
                img = Image.fromarray(frame.astype(np.uint16))
        else:
            raise NotImplementedError('Support only image in categories: color, depth, confidence')
        if width > 0 and height > 0 and frame.shape[0] != height and frame.shape[1] != width:
            img = img.resize((width, height))
        img.save(os.path.join(output_dir, str(frame_name) + image_format))

    def export_frames(self, frames, output_dir, image_format='.png', image_type=ImageCategory.COLOR, scale=1.0):
        if len(frames) == 0:
            return

        if isinstance(frames, dict):
            pass
        elif isinstance(frames, (np.ndarray, list)):
            frames = dict(zip(np.arange(len(frames)), frames))
        else:
            raise NotImplementedError('Export frames only support frames be a dict or a list of frames')
        # cpu_num = cpu_count() if self.cpu_num <= 0 else self.cpu_num
        # cpu_num = min(cpu_count(), cpu_num)

        if self.cpu_num > 1:
            with Pool(processes=self.cpu_num) as pool:
                pool.starmap(StreamDecode.export_frame,
                             [(k, v, output_dir, image_format, self.image_width,
                               self.image_height, image_type, scale, self.colormap) for k, v in
                              frames.items()])
        else:
            for k, v in frames.items():
                StreamDecode.export_frame(k, v, output_dir, image_format, self.image_width,
                                          self.image_height, image_type, scale, self.colormap)
    
    def export_rgb_images(self, output_dir, frame_indices=[], image_format='.png'):
        vr = self.get_video_reader(self.rgb_path)
        if not len(frame_indices):
            frame_indices = list(range(0, min(self.metadata.num_frames, len(vr)), 1))
        chunk_size = min(50, len(frame_indices))
        frame_indices_chunks = np.array_split(frame_indices, len(frame_indices) // chunk_size)
        frame_count = 0
        start = time()
        for frame_indices in frame_indices_chunks:
            frames = self.get_color_frames(vr, frame_indices)
            frame_count += len(frames)
            self.export_frames(dict(zip(frame_indices, frames)), output_dir, image_format)

        log.info(f'Num color frames decoded: {frame_count}')
        log.info(f'Duration decode video: {utils.duration_in_hours(time() - start)}')
        num_frames = len(frame_indices)
        if 0 < num_frames != frame_count or 0 < num_frames != \
                len(io.get_file_list(output_dir, ext=image_format)):
            log.warning(f'Number of frames input {num_frames} does not match the extracted frames')

    def set_depth_fiter_parameters(self, confidence_level=2, delta_threshold=0.05):
        self.confidence_level = confidence_level
        self.delta_threshold = delta_threshold

    def export_depth_images(self, output_dir, frame_indices=[], image_format='.png'):
        start = time()
        if not len(frame_indices):
            frame_indices = list(range(0, self.metadata.num_frames, 1))
        depth_decode_cfg = self.depth_decode_config(frame_indices)

        confidence_filter_flag = self.confidence_path and utils.is_non_zero_file(
            self.confidence_path) and self.confidence_level > 0
        if confidence_filter_flag:
            confidence_decode_cfg = self.confidence_decode_config(frame_indices)

        delta_filter_flag = self.delta_threshold > 0
        scale = 1000.0 if self.metadata.depth_format == 'm' else 1.0
        while depth_decode_cfg.buffer and not depth_decode_cfg.finished:
            # pdb.set_trace()
            if depth_decode_cfg.idx + 1 < len(frame_indices):
                self.step = frame_indices[depth_decode_cfg.idx + 1] - frame_indices[depth_decode_cfg.idx]
            self._decode_stream(depth_decode_cfg, delta_filter=delta_filter_flag)

            if self.step > 1 and delta_filter_flag:
                depth_decode_cfg.prev_frame_flag = ~depth_decode_cfg.prev_frame_flag
            depth_decode_cfg.frame_byte_offset = depth_decode_cfg.frame_index * depth_decode_cfg.frame_size

            depth_decode_cfg.bytes_so_far += depth_decode_cfg.chunk_bytes
            depth_decode_cfg.buffer = depth_decode_cfg.f.read(depth_decode_cfg.chunk_size)

            tmp_frame_indices = depth_decode_cfg.frame_indices[
                                depth_decode_cfg.idx - len(depth_decode_cfg.frames):depth_decode_cfg.idx]
            if confidence_filter_flag:
                self._decode_stream(confidence_decode_cfg, delta_filter=False)
                confidence_decode_cfg.frame_byte_offset = \
                    confidence_decode_cfg.frame_index * confidence_decode_cfg.frame_size
                confidence_decode_cfg.bytes_so_far += confidence_decode_cfg.chunk_bytes
                confidence_decode_cfg.buffer = confidence_decode_cfg.f.read(confidence_decode_cfg.chunk_size)

                filtered_depth_frames = []
                for i in range(len(depth_decode_cfg.frames)):
                    filtered_depth_frame = self.filter_depth_frame(depth_decode_cfg.frames[i],
                                                                   confidence_frame=confidence_decode_cfg.frames[i],
                                                                   confidence_level=self.depth_cfg.confidence_level,
                                                                   delta_threshold=self.depth_cfg.delta_threshold)
                    filtered_depth_frames.append(filtered_depth_frame)

                named_frames = dict(zip(tmp_frame_indices, filtered_depth_frames))
            else:
                named_frames = dict(zip(tmp_frame_indices, depth_decode_cfg.frames))

            # if not (depth_decode_cfg.buffer and not depth_decode_cfg.finished):
            #     pdb.set_trace()
            

            self.export_frames(named_frames, output_dir, image_format=image_format, image_type=ImageCategory.DEPTH,
                               scale=scale)
            depth_decode_cfg.frames.clear()
            if confidence_filter_flag:
                pdb.set_trace()
                confidence_decode_cfg.frames.clear()
        # pdb.set_trace()
        num_frames = len(frame_indices)
        if 0 < num_frames != len(io.get_file_list(output_dir, ext=image_format)):
            log.warning(f'Number of frames input {num_frames} does not match the extracted frames')
        log.info(f'Duration decode depth stream: {utils.duration_in_hours(time() - start)}')

    def export_confidence_images(self, output_dir, frame_indices=[], image_format='.png'):
        start = time()
        confidence_decode_cfg = self.confidence_decode_config(self.stream_path)

        while confidence_decode_cfg.buffer and not confidence_decode_cfg.finished:
            self.step = frame_indices[confidence_decode_cfg.idx + 1] - frame_indices[confidence_decode_cfg.idx]
            self._decode_stream(confidence_decode_cfg)
            confidence_decode_cfg.frame_byte_offset = \
                confidence_decode_cfg.frame_index * confidence_decode_cfg.frame_size

            confidence_decode_cfg.bytes_so_far += confidence_decode_cfg.chunk_bytes
            confidence_decode_cfg.buffer = confidence_decode_cfg.f.read(confidence_decode_cfg.chunk_size)
            named_frames = dict(
                zip(confidence_decode_cfg.frame_indices[
                    confidence_decode_cfg.idx - len(confidence_decode_cfg.frames):confidence_decode_cfg.idx],
                    confidence_decode_cfg.frames))
            self.export_frames(named_frames, output_dir, image_format=image_format,
                               image_type=ImageCategory.CONFIDENCE, scale=0.5)
            confidence_decode_cfg.frames.clear()

        num_frames = len(frame_indices)
        if 0 < num_frames != len(io.get_file_list(output_dir, ext=image_format)):
            log.warning(f'Number of frames input {num_frames} does not match the extracted frames')
        log.info(f'Duration decode confidence stream: {utils.duration_in_hours(time() - start)}')

    def depth_decode_config(self, frame_indices):
        log.info(f'Num depth frames: {len(frame_indices)}')

        depth_chunk_size = 4096
        df = open(self.depth_path, 'rb')
        pixel_size = 2
        depth_frame_size = self.metadata.depth_width * self.metadata.depth_height * pixel_size
        depth_decode_cfg = {
            'idx': 0,
            'frames': [],
            'prev_frame_flag': False,
            'frame': None,
            'prev_frame': None,
            'chunk_size': depth_chunk_size,
            'f': df,
            'decompressor': zlib.decompressobj(-zlib.MAX_WBITS),
            'frame_size': depth_frame_size,
            'bytes_so_far': 0,
            'remain_chunk_bytes': None,
            'frames_batch': self.frames_batch,
            'frame_indices': frame_indices,
            'frame_index': frame_indices[0],
            'buffer': df.read(depth_chunk_size),
            'frame_byte_offset': frame_indices[0] * depth_frame_size,
            'chunk_bytes': 0,
            'finished': False,
            'dtype': 'float16',
        }
        depth_decode_cfg = SimpleNamespace(**depth_decode_cfg)
        return depth_decode_cfg

    def confidence_decode_config(self, frame_indices):
        log.info(f'Num confidence frames: {len(frame_indices)}')

        confidence_chunk_size = 64
        cf = open(self.confidence_path, 'rb')
        pixel_size = 1
        confidence_frame_size = self.metadata.depth_width * self.metadata.depth_height * pixel_size
        confidence_decode_cfg = {
            'idx': 0,
            'frames': [],
            'frame': None,
            'chunk_size': confidence_chunk_size,
            'f': cf,
            'decompressor': zlib.decompressobj(-zlib.MAX_WBITS),
            'frame_size': confidence_frame_size,
            'bytes_so_far': 0,
            'remain_chunk_bytes': None,
            'frames_batch': self.frames_batch,
            'frame_indices': frame_indices,
            'frame_index': frame_indices[0],
            'buffer': cf.read(confidence_chunk_size),
            'frame_byte_offset': frame_indices[0] * confidence_frame_size,
            'chunk_bytes': 0,
            'finished': False,
            'dtype': 'uint8',
        }
        confidence_decode_cfg = SimpleNamespace(**confidence_decode_cfg)
        return confidence_decode_cfg

    @staticmethod
    def get_video_reader(video_path):
        start = time()
        log.info(f'Opening video {video_path} ...')
        vr = VideoReader(video_path, ctx=VideoCPU(0))
        log.info(f'Duration open video: {utils.duration_in_hours(time() - start)}')
        return vr

    @staticmethod
    def get_color_frames(vr, frame_indices):
        frames = vr.get_batch(frame_indices).asnumpy()
        return frames

    @staticmethod
    def filter_depth_frame(depth_frame, confidence_frame=None, prev_depth_frame=None, confidence_level=2,
                           delta_threshold=0.05):
        output_depth_frame = np.copy(depth_frame)
        if prev_depth_frame is not None and prev_depth_frame.ndim and prev_depth_frame.size and delta_threshold > 0:
            delta = np.abs(prev_depth_frame - depth_frame)
            output_depth_frame[delta > delta_threshold] = 0

        if confidence_frame is not None and confidence_level > 0:
            output_depth_frame[confidence_frame < confidence_level] = 0
        return output_depth_frame

    def _decode_stream(self, decode_cfg, delta_filter=False):
        while decode_cfg.buffer:
            tmp_remain_chunk_bytes = None
            stream_chunk = decode_cfg.decompressor.decompress(decode_cfg.buffer)
            decode_cfg.chunk_bytes = len(stream_chunk)
            assert decode_cfg.chunk_bytes < decode_cfg.frame_size, 'decoded chunk size should be less than 1 frame size'
            if decode_cfg.frame_byte_offset <= decode_cfg.bytes_so_far + decode_cfg.chunk_bytes:
                skip_bytes = decode_cfg.frame_byte_offset - decode_cfg.bytes_so_far
                if decode_cfg.remain_chunk_bytes is None:
                    frame_bytes = deque(stream_chunk[skip_bytes:], maxlen=decode_cfg.frame_size)
                else:
                    frame_bytes = decode_cfg.remain_chunk_bytes
                    num_bytes = min(len(stream_chunk), decode_cfg.frame_size - len(frame_bytes))
                    frame_bytes.extend(stream_chunk[:num_bytes])
                    if len(frame_bytes) == decode_cfg.frame_size:
                        tmp_remain_chunk_bytes = deque(stream_chunk[num_bytes:],
                                                       maxlen=decode_cfg.frame_size)

                while len(frame_bytes) < decode_cfg.frame_size:
                    stream_chunk = decode_cfg.decompressor.decompress(decode_cfg.f.read(decode_cfg.chunk_size))
                    decode_cfg.bytes_so_far += len(stream_chunk)
                    assert len(stream_chunk) < decode_cfg.frame_size, \
                        'decoded chunk size should be less than 1 frame size'
                    num_bytes = min(len(stream_chunk), decode_cfg.frame_size - len(frame_bytes))
                    frame_bytes.extend(stream_chunk[:num_bytes])
                    if len(frame_bytes) + len(stream_chunk) > decode_cfg.frame_size:
                        tmp_remain_chunk_bytes = deque(stream_chunk[num_bytes:],
                                                       maxlen=decode_cfg.frame_size)

                if delta_filter and self.step > 1 and decode_cfg.prev_frame_flag:
                    decode_cfg.prev_frame = np.frombuffer(bytes(frame_bytes), dtype=decode_cfg.dtype). \
                        reshape(self.metadata.depth_height, self.metadata.depth_width)
                    assert np.all((decode_cfg.prev_frame >= 0)), 'decoded frame should not have negative values'

                    if decode_cfg.frame_index + 1 == decode_cfg.frame_indices[decode_cfg.idx]:
                        decode_cfg.remain_chunk_bytes = tmp_remain_chunk_bytes

                    decode_cfg.frame_index = decode_cfg.frame_indices[decode_cfg.idx]
                else:
                    if delta_filter and self.step == 1:
                        decode_cfg.prev_frame = np.copy(decode_cfg.frame)

                    decode_cfg.frame = np.frombuffer(bytes(frame_bytes), dtype=decode_cfg.dtype). \
                        reshape(self.metadata.depth_height, self.metadata.depth_width)
                    assert np.all((decode_cfg.frame >= 0)), 'decoded frame should not have negative values'
                    if delta_filter:
                        filtered_depth_frame = self.filter_depth_frame(decode_cfg.frame,
                                                                       confidence_frame=None,
                                                                       prev_depth_frame=decode_cfg.prev_frame,
                                                                       confidence_level=-1,
                                                                       delta_threshold=self.depth_cfg.delta_threshold)
                        decode_cfg.frames.append(filtered_depth_frame)
                    else:
                        decode_cfg.frames.append(decode_cfg.frame)

                    decode_cfg.idx += 1
                    if decode_cfg.idx + 1 < len(decode_cfg.frame_indices):
                        self.step = decode_cfg.frame_indices[decode_cfg.idx + 1] - decode_cfg.frame_indices[decode_cfg.idx]

                    if decode_cfg.idx >= len(decode_cfg.frame_indices):
                        decode_cfg.finished = True
                        break

                    current_frame_index = decode_cfg.frame_index
                    if delta_filter and self.step > 1:
                        decode_cfg.frame_index = decode_cfg.frame_indices[decode_cfg.idx] - 1
                    else:
                        decode_cfg.frame_index = decode_cfg.frame_indices[decode_cfg.idx]

                    if current_frame_index + 1 == decode_cfg.frame_index:
                        decode_cfg.remain_chunk_bytes = tmp_remain_chunk_bytes
                    else:
                        decode_cfg.remain_chunk_bytes = None

                    if decode_cfg.idx % decode_cfg.frames_batch == 0:
                        break

                if delta_filter and self.step > 1:
                    decode_cfg.prev_frame_flag = ~decode_cfg.prev_frame_flag
                decode_cfg.frame_byte_offset = decode_cfg.frame_index * decode_cfg.frame_size

            decode_cfg.bytes_so_far += decode_cfg.chunk_bytes
            decode_cfg.buffer = decode_cfg.f.read(decode_cfg.chunk_size)

    
    