import cv2
import numpy as np
import matplotlib.pyplot as plt

class AvgRecorder(object):
    """
    Average and current value recorder
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def duration_in_hours(duration):
    t_m, t_s = divmod(duration, 60)
    t_h, t_m = divmod(t_m, 60)
    duration_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    return duration_time

def get_pose(obb):
    front = np.asarray(obb['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(obb['up'])
    up = up / np.linalg.norm(up)
    right = np.cross(up, front)
    orientation = np.column_stack([-right, up, -front]).flatten(order='F')
    return orientation

def crop_image(im, o_widith, o_height):
    width, height = im.size  # Get dimensions

    left = (width - o_widith) / 2
    top = (height - o_height) / 2
    right = (width + o_widith) / 2
    bottom = (height + o_height) / 2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def image_colormap(image, colormap='jet', mask=None, clamp=None):
    cm = plt.get_cmap(colormap)
    if clamp:
        color = cm(np.asarray(image).astype('float') / clamp)
    else:
        color = cm(np.asarray(image).astype('float'))
    color *= 255.0
    color = color.astype('uint8')
    if mask is not None:
        color = cv2.bitwise_and(color, color, mask=mask.astype('uint8'))
    color = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
    return color