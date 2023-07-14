import os
import sys
import pdb
import time
import json
import logging
import numpy as np
import pyrender
import trimesh
import cv2

from OpenGL.GL import *
from plyfile import PlyData, PlyElement
from pyrender.constants import RenderFlags

from multiscan.utils import io
from core import Mask2DCore

from pyrender import Renderer, OffscreenRenderer
os.environ['PYOPENGL_PLATFORM'] = 'egl'

log = logging.getLogger(__name__)

class MyRenderer(Renderer):
    def __init__(self, viewport_width, viewport_height, point_size=1.0):
        super().__init__(viewport_width, viewport_height, point_size)
    
    def read_color_buf(self):
        """Read and return the current viewport's color buffer.

        Alpha cannot be computed for an on-screen buffer.

        Returns
        -------
        color_im : (h, w, 3) float32
            The color buffer in RGB float format.
        """
        # Extract color image from frame buffer
        width, height = self.viewport_width, self.viewport_height
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glReadBuffer(GL_FRONT)
        color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_SHORT)
        print('a')

        # Re-format them into numpy arrays
        color_im = np.frombuffer(color_buf, dtype=np.float32)
        color_im = color_im.reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        # Resize for macos if needed
        if sys.platform == 'darwin':
            color_im = self._resize_image(color_im, True)

        return color_im

    def _read_main_framebuffer(self, scene, flags):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]

        # Bind framebuffer and blit buffers
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
        glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height,
            GL_COLOR_BUFFER_BIT, GL_LINEAR
        )
        glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height,
            GL_DEPTH_BUFFER_BIT, GL_NEAREST
        )
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)

        # Read depth
        depth_buf = glReadPixels(
            0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT
        )
        depth_im = np.frombuffer(depth_buf, dtype=np.float32)
        depth_im = depth_im.reshape((height, width))
        depth_im = np.flip(depth_im, axis=0)
        inf_inds = (depth_im == 1.0)
        depth_im = 2.0 * depth_im - 1.0
        z_near = scene.main_camera_node.camera.znear
        z_far = scene.main_camera_node.camera.zfar
        noninf = np.logical_not(inf_inds)
        if z_far is None:
            depth_im[noninf] = 2 * z_near / (1.0 - depth_im[noninf])
        else:
            depth_im[noninf] = ((2.0 * z_near * z_far) /
                                (z_far + z_near - depth_im[noninf] *
                                (z_far - z_near)))
        depth_im[inf_inds] = 0.0

        # Resize for macos if needed
        if sys.platform == 'darwin':
            depth_im = self._resize_image(depth_im)

        if flags & RenderFlags.DEPTH_ONLY:
            return depth_im

        # Read color
        if flags & RenderFlags.RGBA:
            color_buf = glReadPixels(
                0, 0, width, height, GL_RGBA, GL_UNSIGNED_SHORT
            )
            color_im = np.frombuffer(color_buf, dtype=np.uint16)
            color_im = color_im.reshape((height, width, 4))
        else:
            color_buf = glReadPixels(
                0, 0, width, height, GL_RGB, GL_UNSIGNED_SHORT
            )
            color_im = np.frombuffer(color_buf, dtype=np.uint16)
            color_im = color_im.reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        # Resize for macos if needed
        if sys.platform == 'darwin':
            color_im = self._resize_image(color_im, True)

        return color_im, depth_im

    def _configure_main_framebuffer(self):
        # If mismatch with prior framebuffer, delete it
        if (self._main_fb is not None and
                self.viewport_width != self._main_fb_dims[0] or
                self.viewport_height != self._main_fb_dims[1]):
            self._delete_main_framebuffer()

        # If framebuffer doesn't exist, create it
        if self._main_fb is None:
            # Generate standard buffer
            self._main_cb, self._main_db = glGenRenderbuffers(2)

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_RGBA16,
                self.viewport_width, self.viewport_height
            )

            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db)
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                self.viewport_width, self.viewport_height
            )

            self._main_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_RENDERBUFFER, self._main_cb
            )
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER, self._main_db
            )

            # Generate multisample buffer
            self._main_cb_ms, self._main_db_ms = glGenRenderbuffers(2)
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb_ms)
            glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, 0, GL_RGBA16,
                self.viewport_width, self.viewport_height
            )
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db_ms)
            glRenderbufferStorageMultisample(
                GL_RENDERBUFFER, 0, GL_DEPTH_COMPONENT24,
                self.viewport_width, self.viewport_height
            )
            self._main_fb_ms = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb_ms)
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_RENDERBUFFER, self._main_cb_ms
            )
            glFramebufferRenderbuffer(
                GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER, self._main_db_ms
            )

            self._main_fb_dims = (self.viewport_width, self.viewport_height)

class MyOffscreenRenderer(OffscreenRenderer):
    def __init__(self, viewport_width, viewport_height, point_size=1.0):
        super().__init__(viewport_width, viewport_height, point_size)

    def _create(self):
        if 'PYOPENGL_PLATFORM' not in os.environ:
            from pyrender.platforms.pyglet_platform import PygletPlatform
            self._platform = PygletPlatform(self.viewport_width,
                                            self.viewport_height)
        elif os.environ['PYOPENGL_PLATFORM'] == 'egl':
            from pyrender.platforms import egl
            device_id = int(os.environ.get('EGL_DEVICE_ID', '0'))
            egl_device = egl.get_device_by_index(device_id)
            self._platform = egl.EGLPlatform(self.viewport_width,
                                             self.viewport_height,
                                             device=egl_device)
        elif os.environ['PYOPENGL_PLATFORM'] == 'osmesa':
            from pyrender.platforms.osmesa import OSMesaPlatform
            self._platform = OSMesaPlatform(self.viewport_width,
                                            self.viewport_height)
        else:
            raise ValueError('Unsupported PyOpenGL platform: {}'.format(
                os.environ['PYOPENGL_PLATFORM']
            ))
        self._platform.init_context()
        self._platform.make_current()
        self._renderer = MyRenderer(self.viewport_width, self.viewport_height)
    

class Project2D(Mask2DCore):
    def __init__(self):
        super().__init__()
        self.plymesh = None
        self.objmesh = None
        self.camera_nodes = []

        self.znear = 0.001
        self.zfar = 1000

        self.width = 256
        self.height = 192
        
        self._bg_color = np.asarray([0.0, 0.0, 0.0, 0.0])
        self.scene = None
        self.debug = True

    def set_cameras(self, cameras: list):
        self.cameras = cameras

    def load_obj(self, obj_path, transform=np.eye(4)):
        self.objmesh = trimesh.load(obj_path, force='scene')
        self.objmesh.apply_transform(transform)

    def load_ply(self, ply_path, transform=np.eye(4)):
        plydata = PlyData.read(ply_path)
        x = np.asarray(plydata['vertex']['x'])
        y = np.asarray(plydata['vertex']['y'])
        z = np.asarray(plydata['vertex']['z'])
        # nx = np.asarray(plydata['vertex']['nx'])
        # ny = np.asarray(plydata['vertex']['ny'])
        # nz = np.asarray(plydata['vertex']['nz'])
        vertices = np.column_stack((x, y, z))
        # vertex_normals = np.column_stack((nx, ny, nz))
        triangles = np.vstack(plydata['face'].data['vertex_indices'])
        object_ids = plydata['face'].data['objectId']
        part_ids = plydata['face'].data['partId']
        column = np.ones(len(triangles))
        bit_max = 2**16 - 1.0
        face_colors = np.column_stack((object_ids / bit_max, part_ids / bit_max, column, column))
        
        positions = vertices[triangles].reshape((3 * len(triangles), 3))
        colors = np.repeat(face_colors, 3, axis=0)
        
        primitive = pyrender.Primitive(positions=positions, color_0=colors, material=None, mode=4, poses=[transform])
        self.plymesh = pyrender.Mesh([primitive])

    def set_window_size(self, width, height):
        self.width = width
        self.height = height

    def _initialize_render_cameras(self):
        for camera in self.cameras:
            fx = camera.intrinsics[0, 0]
            fy = camera.intrinsics[1, 1]
            cx = camera.intrinsics[0, 2]
            cy = camera.intrinsics[1, 2]
            render_camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=0.001, zfar=1000.0)

            pose = np.linalg.inv(camera.extrinsics)
            node = pyrender.Node(name=camera.name, camera=render_camera, matrix=pose)
            self.camera_nodes.append(node)

    def export_mask_images(self, output_dir, ext='.png'):
        self.scene = pyrender.Scene(bg_color=self._bg_color)
        self.scene.ambient_light = [1.0, 1.0, 1.0]
        self.scene.add(self.plymesh)

        renderer = MyOffscreenRenderer(viewport_width=self.width, viewport_height=self.height)
        flags = RenderFlags.FLAT
        for camera_node in self.camera_nodes:
            self.scene.add_node(camera_node)

            mask, _ = renderer.render(self.scene, flags)
            self.scene.remove_node(camera_node)
            # pdb.set_trace()
            cv2.imwrite(os.path.join(output_dir, camera_node.name+ext), mask)
    
    def export_rgb_images(self, output_dir, ext='.png'):
        # pdb.set_trace()
        self.scene = pyrender.Scene.from_trimesh_scene(self.objmesh, bg_color=self._bg_color)

        renderer = OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)
        flags = RenderFlags.FLAT
        for camera_node in self.camera_nodes:
            self.scene.add_node(camera_node)

            img_rgb, _ = renderer.render(self.scene, flags)
            self.scene.remove_node(camera_node)
            # pdb.set_trace()
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, camera_node.name+ext), img_rgb)
            # pdb.set_trace()

    def export_depth_images(self):
        pass
        




