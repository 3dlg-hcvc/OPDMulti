import cv2
import os
import re
import json
import h5py
import glob
import shutil
import trimesh
import resource
import subprocess as subp
import sys
import datetime
import traceback
from enum import Enum
import multiprocessing
import multiprocessing.pool
from functools import partial
from tqdm.contrib.concurrent import process_map
from pycococreatortools import pycococreatortools

from re import T
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from plyfile import PlyData
import matplotlib.pyplot as plt
from dataclasses import dataclass


def file_exist(file_path, ext=''):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return False
    elif ext in os.path.splitext(file_path)[1] or not ext:
        return True
    return False


def is_non_zero_file(file_path):
    return True if os.path.isfile(file_path) and os.path.getsize(file_path) > 0 else False


def file_extension(file_path):
    return os.path.splitext(file_path)[1]


def folder_exist(folder_path):
    if not os.path.exists(folder_path) or os.path.isfile(folder_path):
        return False
    else:
        return True


def ensure_dir_exists(path):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
    except OSError:
        raise


def make_clean_folder(path_folder):
    try:
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        else:
            shutil.rmtree(path_folder)
            os.makedirs(path_folder)
    except OSError:
        if not os.path.isdir(path_folder):
            raise


def sorted_alphanum(file_list):
    """sort the file list by arrange the numbers in filenames in increasing order

    :param file_list: a file list
    :return: sorted file list
    """
    if len(file_list) <= 1:
        return file_list, [0]

    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]

    indices = [i[0]
               for i in sorted(enumerate(file_list), key=lambda x: alphanum_key(x[1]))]
    return sorted(file_list, key=alphanum_key), indices

def get_file_list(path, ext='', join_path=True):
    file_list = []
    if not os.path.exists(path):
        return file_list

    for filename in os.listdir(path):
        file_ext = file_extension(filename)
        if (ext in file_ext or not ext) and os.path.isfile(os.path.join(path, filename)):
            if join_path:
                file_list.append(os.path.join(path, filename))
            else:
                file_list.append(filename)
    file_list, _ = sorted_alphanum(file_list)
    return file_list

def get_folder_list(path, join_path=True):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    folder_list = []
    for foldername in os.listdir(path):
        if not os.path.isdir(os.path.join(path, foldername)):
            continue
        if join_path:
            folder_list.append(os.path.join(path, foldername))
        else:
            folder_list.append(foldername)
    folder_list, _ = sorted_alphanum(folder_list)
    return folder_list


def filesize(file_path):
    if os.path.isfile(file_path):
        return os.path.getsize(file_path)
    else:
        return 0


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def write_json(data, filename, indent=2):
    if folder_exist(os.path.dirname(filename)):
        with open(filename, "w+") as fp:
            json.dump(data, fp, indent=indent)
    if not file_exist(filename):
        raise OSError('Cannot create file {}!'.format(filename))


def read_json(filename):
    if file_exist(filename):
        with open(filename, "r") as fp:
            data = json.load(fp)
        return data

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # print(f'memory limit {soft} TO {hard}')
    # print(f'memory limit set to {maxsize} GB')
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

# TODO: make memory limitation configurable
def call(cmd, log, rundir='', env=None, desc=None, cpu_num=0, mem=48, print_at_run=True, test_mode=False):
    if not cmd:
        log.warning('No command given')
        return 0
    if test_mode:
        log.info('Running ' + str(cmd))
        return -1
    cwd = os.getcwd()
    res = -1
    prog = None

    # constraint cpu usage with taskset
    if cpu_num > 0:
        all_cpus = list(range( min(psutil.cpu_count(), cpu_num)))
        sub_cpus = all_cpus[:cpu_num]
        str_cpus = ','.join(str(e) for e in sub_cpus)
        taskset_cmd = ['taskset', '-c', str_cpus]
        cmd = taskset_cmd + cmd
    
    try:
        start_time = timer()
        if rundir:
            os.chdir(rundir)
            log.info('Currently in ' + os.getcwd())
        log.info('Running ' + str(cmd))
        log.info(f'memory limit set to {mem} GB')
        setlimits = lambda: limit_memory(mem*1000*1000*1000) # in GB
        prog = subp.Popen(cmd, stdout=subp.PIPE, stderr=subp.STDOUT, env=env, preexec_fn=setlimits)
        # print output during the running
        if print_at_run:
            while True:
                nextline = prog.stdout.readline()
                if nextline == b'' and prog.poll() is not None:
                    break
                sys.stdout.write(nextline.decode("utf-8"))
                sys.stdout.flush()
        
        out, err = prog.communicate()
        if out:
            log.info(out.decode("utf-8"))
        if err:
            log.error('Errors reported running ' + str(cmd))
            log.error(err.decode("utf-8"))
        end_time = timer()
        delta_time = end_time - start_time
        desc_str = desc + ', ' if desc else ''
        desc_str = desc_str + 'cmd="' + str(cmd) + '"'
        log.info('Time=' + str(datetime.timedelta(seconds=delta_time)) + ' for ' + desc_str)
        res = prog.returncode
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt")
    except Exception as e:
        if prog is not None:
            prog.kill()
            out, err = prog.communicate()
        log.error(traceback.format_exc())
    os.chdir(cwd)
    return res

# https://stackoverflow.com/questions/3431825/generating-a-md5-checksum-of-a-file
def md5(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(blocksize), b''):
            hash.update(chunk)
    return hash.hexdigest()

# http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def natural_size(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class TIMEOUT(Enum):
    SECOND = 1
    MINUTE = 60
    HOUR = 3600

# reference https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(multiprocessing.pool.Pool):
    _wrap_exception = True

    def Process(self, *args, **kwds):
        proc = super(NoDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc
        

# set mem limit for program
# reference https://stackoverflow.com/questions/41105733/limit-ram-usage-to-python-program
def set_memory_limit(percentage: float):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory_limit(percentage=0.8):
    def decorator(function):
        def wrapper(*args, **kwargs):
            set_memory_limit(percentage)
            try:
                return function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 /1024
                sys.stderr.write('\n\nERROR: Memory Exception, remaining memory  %.2f GB\n' % mem)
                sys.exit(1)
        return wrapper
    return decorator


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


def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# Unuseful Information
def get_info():

    info = {
        "description": "MultiScan",
        "url": "",
        "version": "0.0.0",
        "year": 2022,
        "contributor": "Xiaohao, Shawn, Angel",
        "date_created": "2022/07/27"
    }
    return info

# Unuseful Information
def get_licenses():
    licenses = [
    {
        "url": "",
        "id": 1,
        "name": ""
    }]

    return licenses


# MotionNet Version
def get_categories(cat_id):
    categories = []

    # Don't make use of super category here
    supercategory = ['Container_parts']

    # Drawer
    info = {}
    info["supercategory"] = supercategory[0]
    info["id"] = 1
    info["name"] = "drawer"
    categories.append(info)
    cat_id['drawer'] = 1
    # Door
    info = {}
    info["supercategory"] = supercategory[0]
    info["id"] = 2
    info["name"] = "door"
    categories.append(info)
    cat_id['door'] = 2
    # Lid
    info = {}
    info["supercategory"] = supercategory[0]
    info["id"] = 3
    info["name"] = "lid"
    categories.append(info)
    cat_id['lid'] = 3

    return categories

# MotionNet Version
def get_images(img_folder, annotation_folder, img_name_id, motion_annotations, height=192, width=256):
    images = []

    img_id = 1
    for img_file_name in os.listdir(img_folder):
        img_name = img_file_name.split('.')[0]

        # Read corresponding annotation file
        annotation_file = open(f'{annotation_folder}/{img_name}.json')
        data = json.load(annotation_file)
        annotation_file.close()
        
        img_path = os.path.join(img_folder, img_file_name)
        image = {
                "license": 1,
                "file_name": img_file_name,
                "coco_url" :"",
                "height": height,
                "width" : width,
                "date_captured": "",
                "flickr_url": "",
                "id": img_id
            }

        img_name_id[img_name] = img_id

        # MotionNet Extra Annotation
        # Add corresponding depth image name
        image['depth_file_name'] = img_name + '_d.png'
        # Add camera parameters to the image
        camera = {
            "intrinsic": data["intrinsics"],
            "extrinsic": data["extrinsics"],
        }
        image['camera'] = camera
        # Add model label
        # image['label'] = data['label']

        images.append(image)

        # Preprocess the motion annotations for adding it into the masks
        motion_annotations[img_name] = {}
        for motion in data['articulation']:
            motion_annotations[img_name][motion['partId']] = motion

        img_id += 1

    return images

# MotionNet Version
def get_annotation_info(seg_id, img_id, category_info, mask_path, motion):
    
    mask = Image.open(mask_path)
    binary_mask = np.asarray(mask.convert('L')).astype(np.uint8)

    for i in range(np.shape(binary_mask)[0]):
        for j in range(np.shape(binary_mask)[1]):
            if(binary_mask[i, j] > 1):
                binary_mask[i, j] = 1
    
    annotation_info = pycococreatortools.create_annotation_info(
                        seg_id, img_id, category_info, binary_mask,
                        mask.size, tolerance=2)

    if(annotation_info != None):
        annotation_info['motion'] = motion


    return annotation_info

# MotionNet Version
def get_annotations(img_folder, mask_folder, img_name_id, motion_annotations, cat_id):
    annotations = []
    
    pool = multiprocessing.Pool(processes = 16)

    seg_id = 1
    annotation_infos = []

    for mask_name in os.listdir(mask_folder):
        img_name = mask_name.rsplit('_', 1)[0]
        if img_name in img_name_id.keys():
            img_id = img_name_id[img_name]
        else:
            pdb.set_trace()
            print('ERROR: cannot find image ', img_name)
            exit()
        if int(mask_name.split("_")[-1].split(".")[0]) == 0:
            continue
        part_id = int((mask_name.split('.')[0]).split('_')[-1])
        motion = motion_annotations[img_name][part_id]
        
        if motion['part_label'].strip() not in cat_id:
            print(motion['part_label'].strip())
        part_cat = cat_id[motion['part_label'].strip()]
        
        category_info = {'id': part_cat, 'is_crowd': 0, "object_key": motion["object_key"],}
        mask_path = os.path.join(mask_folder, mask_name)

        annotation_infos.append(pool.apply_async(get_annotation_info, (seg_id, img_id, category_info, mask_path, motion, )))

        seg_id += 1
    
    pool.close()
    pool.join()

    for i in annotation_infos:
        annotation_info = i.get()
        if(annotation_info != None):
            annotations.append(annotation_info)

    return annotations

def save_json(data, save_path):
    out_json = json.dumps(data, sort_keys=True, indent=4, separators=(',', ':'),
                          ensure_ascii=False)
    fo = open(save_path, "w")
    fo.write(out_json)
    fo.close()


def convert_coco(input_dir):
    PROCESSDATAPATH = input_dir + "/all/"

    annotation_data_path = PROCESSDATAPATH + '/coco_annotation/'
    existDir(annotation_data_path)

    # Deal with train data
    print('process train data ...')
    train_origin_path = PROCESSDATAPATH + '/train/rgb'
    train_mask_path = PROCESSDATAPATH + '/train/mask'
    train_annotation_path = PROCESSDATAPATH + '/train/annotation'
    img_name_id = {}
    motion_annotations = {}
    cat_id = {}

    output = {}
    output["info"] = get_info()
    output["licenses"] = get_licenses()
    output["categories"] = get_categories(cat_id)
    output["images"] = get_images(train_origin_path, train_annotation_path, img_name_id, motion_annotations)
    output["annotations"] = get_annotations(train_origin_path, train_mask_path, img_name_id, motion_annotations, cat_id)

    save_json(output, annotation_data_path + 'MotionNet_train.json')

    print('process valid data ...')
    valid_origin_path = PROCESSDATAPATH + 'valid/rgb'
    valid_mask_path = PROCESSDATAPATH + 'valid/mask'
    valid_annotation_path = PROCESSDATAPATH + 'valid/annotation'

    output = {}
    output["info"] = get_info()
    output["licenses"] = get_licenses()
    output["categories"] = get_categories(cat_id)
    output["images"] = get_images(valid_origin_path, valid_annotation_path, img_name_id, motion_annotations)
    output["annotations"] = get_annotations(valid_origin_path, valid_mask_path, img_name_id, motion_annotations, cat_id)

    save_json(output, annotation_data_path + 'MotionNet_valid.json')

    print('process test data ...')
    test_origin_path = PROCESSDATAPATH + 'test/rgb'
    test_mask_path = PROCESSDATAPATH + 'test/mask'
    test_annotation_path = PROCESSDATAPATH + 'test/annotation'

    output = {}
    output["info"] = get_info()
    output["licenses"] = get_licenses()
    output["categories"] = get_categories(cat_id)
    output["images"] = get_images(test_origin_path, test_annotation_path, img_name_id, motion_annotations)
    output["annotations"] = get_annotations(test_origin_path, test_mask_path, img_name_id, motion_annotations, cat_id)

    save_json(output, annotation_data_path + 'MotionNet_test.json')


def final_process(input_dir, output_dir):
    PROCESSPATH = input_dir + "/all/"
    DATASETPATH = output_dir

    # Create the dirs
    dir_names = ['/train/', '/valid/', '/test/', '/annotations/', '/depth/']
    for dir_name in dir_names:
        existDir(DATASETPATH + dir_name)

    # Move the origin images and depth images
    origin_dir = ['/train/', '/valid/', '/test/']
    for dir_name in origin_dir:
        print(f'Copying the {dir_name} images')

        # Move the origin images
        input_path = f'{PROCESSPATH}{dir_name}rgb/'
        output_path = f'{DATASETPATH}{dir_name}'
        # Loop the images
        file_paths = glob.glob(f'{input_path}*')
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            new_file_path = f'{output_path}{file_name}'
            os.system(f'cp {file_path} {new_file_path}')

        # Move the depth images
        input_path = f'{PROCESSPATH}{dir_name}depth/'
        output_path = f'{DATASETPATH}/depth/'
        # Loop the images
        file_paths = glob.glob(f'{input_path}*')
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            new_file_path = f'{output_path}{file_name}'
            os.system(f'cp {file_path} {new_file_path}')

    # Move the annotations
    file_paths = glob.glob(f'{PROCESSPATH}coco_annotation/*')
    for file_path in file_paths:
        print('Copying the coco annotations')
        file_name = file_path.split('/')[-1]
        new_file_path = f'{DATASETPATH}/annotations/{file_name}'
        os.system(f'cp {file_path} {new_file_path}')


def get_frame_id(cfg, scan_id):
    file_path = f"{cfg.input_dir}/{scan_id}/depth/*"
    paths = glob.glob(file_path)
    frame_ids = [tmp.split('/')[-1].split(".")[0] for tmp in paths]
    return frame_ids


def split_mask(mask, articulations, obj_part_info):
    masks = []
    cat_ids = ["drawer", "door", "lid", "cover", "covert", "lid_cover"]
    part_channel = mask[:, :, 1]
    for articulation in articulations:
        part_id = articulation["partId"]
        part_info = obj_part_info.loc[obj_part_info["partID"] == part_id]
        try:
            cat_id = part_info["part_label"].values[0].split(".")[0]
        except:
            continue
        if cat_id in cat_ids:
            idx = np.where(part_channel == part_id)
            tmp_mask = np.zeros(part_channel.shape)
            tmp_mask[idx] = 255
            tmp = {
                "part_id": part_id,
                "mask": tmp_mask.astype(np.uint8)
            }
            masks.append(tmp)
    return masks


def update_annotation(annotation, obj_part_info):
    cat_ids = ["drawer", "door", "lid", "cover", "covert","lid_cover"]
    new_annotation = {
        "intrinsics": annotation["intrinsics"],
        "extrinsics": annotation["extrinsics"],
        "diagonal": annotation["diagonal"]
    }
    arti = []
    for articulation in annotation["articulation"]:
        part_info = obj_part_info.loc[obj_part_info["partID"]
                                      == articulation["partId"]]
        cat_id = part_info["part_label"].values[0].split(".")[0]
        if cat_id in cat_ids:
            if cat_id in ["cover", "lid_cover", "covert", "lid"]:
                cat_id = "lid"
            articulation["part_label"] = cat_id
            arti.append(articulation)

    new_annotation["articulation"] = arti

    return new_annotation

def get_obj_part_info(origin_annotation):
    obj_part_info = pd.DataFrame({"objectID": [], "partID": [], "object_label": [
    ], "part_label": [], "mobility_type": []})
    for object in origin_annotation["objects"]:
        obj_id = object["objectId"]
        obj_label = object["label"]
        mobility_type = object["mobilityType"]
        for part_id in object["partIds"]:
            part_label = origin_annotation["parts"][part_id-1]["label"]
            tmp = pd.DataFrame({"objectID": [obj_id], "partID": [part_id], "object_label": [obj_label],
                                "part_label": [part_label], "mobility_type": [mobility_type]})
            obj_part_info = obj_part_info.append(tmp, ignore_index=True)

    return obj_part_info

@dataclass
class MeshInfo:
    mesh: trimesh.Trimesh
    object_ids: np.ndarray
    part_ids: np.ndarray
    obj_verts_id: np.ndarray
    part_verts_id: np.ndarray


def get_bound(ply_path):
    plydata = PlyData.read(ply_path)
    x = np.asarray(plydata['vertex']['x'])
    y = np.asarray(plydata['vertex']['y'])
    z = np.asarray(plydata['vertex']['z'])
    vertices = np.column_stack((x, y, z))

    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])

    min_bound = [x_min, y_min, z_min]
    max_bound = [x_max, y_max, z_max]

    diameter = np.linalg.norm(np.asarray(min_bound) - np.asarray(max_bound))

    return diameter

def update_articulation(mask, threshold, scan_id, frame_id, annotations, obj_part_info):
    articulation = annotations["articulations"]
    obbs = annotations["obbs"]
    part_channel = mask[:, :, 1]
    part_ids = np.unique(part_channel)
    new_arti_anno = []

    obb_obj_idx = {}
    for idx, obb in enumerate(obbs):
        obb_obj_idx.update({obb["objectId"]: idx})
    
    for index, part in enumerate(articulation):
        if part["partId"] in part_ids:
            try:
                objectID = int(obj_part_info.loc[obj_part_info["partID"] == part["partId"]]["objectID"].values)
            except:
                print("Failed scan ID", scan_id)
                continue

            idx = np.where(part_channel == part["partId"])
            xmin = min(idx[0])
            ymin = min(idx[1])
            xmax = max(idx[0])
            ymax = max(idx[1])

            width = xmax - xmin
            height = ymax - ymin

            bbox = np.asarray([xmin, ymin, width, height]).astype(np.float64)
            articulation[index]["bbox"] = bbox.tolist()
            part_pixel_num = len(idx[0])
            articulation[index]["pixel_num"] = np.float64(part_pixel_num)
            if articulation[index]["pixel_num"] < threshold:
                continue
            articulation[index]["object_key"] = f"{scan_id}_{frame_id}_{objectID}"

            new_arti_anno.append(articulation[index])

    return new_arti_anno


def get_mesh_info(ply_path, alignment):
    plydata = PlyData.read(ply_path)
    x = np.asarray(plydata['vertex']['x'])
    y = np.asarray(plydata['vertex']['y'])
    z = np.asarray(plydata['vertex']['z'])
    vertices = np.column_stack((x, y, z))
    # vertex_normals = np.column_stack((nx, ny, nz))
    triangles = np.vstack(plydata['face'].data['vertex_indices'])
    alignment = np.reshape(
        alignment['coordinate_transform'], (4, 4), order='F')
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=triangles, process=False)
    mesh.apply_transform(alignment)
    object_ids = plydata['face'].data['objectId']
    part_ids = plydata['face'].data['partId']

    # get objectID for each vertex
    obj_verts_id = np.zeros(len(vertices))
    for idx, triangle in enumerate(triangles):
        obj_verts_id[triangle[0]] = obj_verts_id[triangle[1]
                                                 ] = obj_verts_id[triangle[2]] = object_ids[idx]

    # get partID for each vertex
    part_verts_id = np.zeros(len(vertices))
    for idx, triangle in enumerate(triangles):
        part_verts_id[triangle[0]] = part_verts_id[triangle[1]
                                                   ] = part_verts_id[triangle[2]] = part_ids[idx]

    mesh_info = MeshInfo(mesh, object_ids, part_ids,
                         obj_verts_id, part_verts_id)

    return mesh_info


def get_obj_frame_info(cfg, mesh_info, annotations, mask):
    # read plydata and project vertices to image plane
    intrinsics = np.reshape(annotations["intrinsics"], (3, 3), order='F')
    extrinsics = np.reshape(annotations["extrinsics"], (4, 4), order='F')
    mesh = mesh_info.mesh.copy()
    mesh = mesh.apply_transform(extrinsics)
    vertices = mesh.vertices
    v_pixels = np.dot(vertices, intrinsics.T)
    v_pixels /= v_pixels[:, -1].reshape(-1, 1)

    # count vertex in the frame for different object
    obj_frame_info = []
    pixel_x_in_frame = (0 < v_pixels[:, 0]) & (
        v_pixels[:, 0] < cfg.image_width)
    pixel_y_in_frame = (0 < v_pixels[:, 1]) & (
        v_pixels[:, 1] < cfg.image_height)
    obj_in_frame = pixel_x_in_frame & pixel_y_in_frame

    obj_ids = np.unique(mesh_info.object_ids)
    for obj_id in obj_ids:
        if obj_id != 0:
            verts_idx = np.where(mesh_info.obj_verts_id == obj_id)
            valid_obj_verts_num = np.sum(obj_in_frame[verts_idx])
            pixel_idx = np.where(mask[:, :, 0] == obj_id)
            tmp = {
                "object_id": int(obj_id),
                "pixel_count": np.float64(len(pixel_idx[0])),
                "total_vertex_count": np.float64(len(verts_idx[0])),
                "vertex_count": np.float64(valid_obj_verts_num),
            }
            obj_frame_info.append(tmp)

    return obj_frame_info


def get_part_frame_info(cfg, mesh_info, annotations, mask):
    # read plydata and project vertices to image plane
    intrinsics = np.reshape(annotations["intrinsics"], (3, 3), order='F')
    extrinsics = np.reshape(annotations["extrinsics"], (4, 4), order='F')
    mesh = mesh_info.mesh.copy()
    mesh = mesh.apply_transform(extrinsics)
    vertices = mesh.vertices
    v_pixels = np.dot(vertices, intrinsics.T)
    v_pixels /= v_pixels[:, -1].reshape(-1, 1)

    # count vertex in the frame for different object
    part_frame_info = []
    pixel_x_in_frame = (0 < v_pixels[:, 0]) & (
        v_pixels[:, 0] < cfg.image_width)
    pixel_y_in_frame = (0 < v_pixels[:, 1]) & (
        v_pixels[:, 1] < cfg.image_height)
    part_in_frame = pixel_x_in_frame & pixel_y_in_frame

    part_ids = np.unique(mesh_info.part_ids)
    for part_id in part_ids:
        if part_id != 0:
            verts_idx = np.where(mesh_info.part_verts_id == part_id)
            valid_part_verts_num = np.sum(part_in_frame[verts_idx])
            pixel_idx = np.where(mask[:, :, 1] == part_id)
            tmp = {
                "part_id": int(part_id),
                "pixel_count": np.float64(len(pixel_idx[0])),
                "total_vertex_count": np.float64(len(verts_idx[0])),
                "vertex_count": np.float64(valid_part_verts_num),
            }
            part_frame_info.append(tmp)

    return part_frame_info


def get_pose_from_obb(obb):
    if obb.get('front') is None:
        return np.eye(4)
    front = np.asarray(obb['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(obb['up'])
    up = up / np.linalg.norm(up)
    right = np.cross(up, front)
    orientation = np.eye(4)
    orientation[:3, :3] = np.stack([right, up, front], axis=0)
    translation = np.eye(4)
    translation[:3, 3] = -np.asarray(obb['centroid'])
    return np.dot(orientation, translation)


def motion_real_diagonal(motion_anno_path, real_attr_path, real_name_map_path):
    datasets = ["."]

    diagonal_dict = {}
    scan_index = 0
    scan_idx = 0
    real_name_map = {}
    tmp_scanid = []
    for dataset in datasets:
        anno_paths = glob.glob(f"{motion_anno_path}/{dataset}/*.json")
        for anno_path in anno_paths:
            anno_file = open(anno_path)
            anno = json.load(anno_file)
            anno_file.close()
            # if not len(anno.keys()) == 1:
            #     print(f"Something wrong: {anno_path}")
            if list(anno.keys())[0] in diagonal_dict.keys():
                print(f"Something wrong 2: {anno_path} {diagonal_dict[list(anno.keys())[0]]}")
            diagonal_dict[str(scan_index)] = {"diameter": anno["diagonal"]}
            scan_id = anno_path.split("/")[-1].split(".")[0].rsplit("_",1)[0]
            frame_id = anno_path.split("/")[-1].split(".")[0].rsplit("_",1)[1]
            if scan_id not in tmp_scanid:
                real_name_map[scan_id] = str(scan_idx)
                scan_idx += 1
                tmp_scanid.append(scan_id)
            diagonal_dict[str(scan_id)+"-"+str(frame_id)] = {"diameter": anno["diagonal"]}
            scan_index += 1
    
    diagonal_file = open(real_attr_path, 'w')
    json.dump(diagonal_dict, diagonal_file)
    diagonal_file.close()

    map_file = open(real_name_map_path, 'w')
    json.dump(real_name_map, map_file)
    map_file.close()


def change_file_name(dir, name_map, test_ids, train_ids, val_ids, test_scans, val_scans, train_scans, image):
    image_name = (image.split('/')[-1]).split('.')[0]
    scan_id = (image.split('/')[-1]).split('.')[0][:50]
    with open(f"{dir}/annotation/{image_name}.json") as f:
        anno = json.load(f)
    # Make it consistent with 2DMotion dataset
    extrinsic_matrix = np.reshape(
        anno["extrinsics"], (4, 4), order="F").flatten(order="F")
    anno["extrinsics"] = list(extrinsic_matrix)
    with open(f"{dir}/annotation/{image_name}.json", 'w') as f:
        json.dump(anno, f)
    motion_number = len(anno["articulation"])
    motion_ids = [anno["partId"] for anno in anno["articulation"]]
    obj_ids = [int(anno["object_key"].split("_")[-1]) for anno in anno["articulation"]]
    mask_paths = glob.glob(f"{dir}/mask/{image_name}_*")
    obj_mask_paths = glob.glob(f"{dir}/obj_mask/{image_name}_*")
    if len(mask_paths) > 1 or int(mask_paths[0].split("_")[-1].split(".")[0]) != 0:
        if not motion_number == len(mask_paths):
            print(f"Not consistent mask and motion {image}")
    # Rename the RGB
    model_name = image_name.rsplit('-', 1)[0].rsplit("_", 1)[0]
    new_image_name = name_map[model_name] + \
            '-' + image_name.rsplit('_', 1)[1]
    os.system(
        f"mv {dir}/rgb/{image_name}.png {dir}/rgb/{new_image_name}.png")
    # Rename the depth
    os.system(
        f"mv {dir}/depth/{image_name}_d.png {dir}/depth/{new_image_name}_d.png")
    # Rename the annotation
    os.system(
        f"mv {dir}/annotation/{image_name}.json {dir}/annotation/{new_image_name}.json")
    # Rename all the masks
    for mask_path in mask_paths:
        mask_name = (mask_path.split('/')[-1]).split('.')[0]
        if int(mask_name.rsplit('_', 1)[1]) not in motion_ids and int(mask_name.rsplit('_', 1)[1]) != 0:
            import pdb
            pdb.set_trace()
        new_mask_name = f"{new_image_name}_{mask_name.rsplit('_', 1)[1]}"
        os.system(
            f"mv {dir}/mask/{mask_name}.png {dir}/mask/{new_mask_name}.png")
    
    # add obj mask operation
    for mask_path in obj_mask_paths:
        mask_name = (mask_path.split('/')[-1]).split('.')[0]
        if int(mask_name.rsplit('_', 1)[1]) not in obj_ids and int(mask_name.rsplit('_', 1)[1]) != 0:
            import pdb
            pdb.set_trace()
        new_mask_name = f"{new_image_name}_{mask_name.rsplit('_', 1)[1]}"
        os.system(
            f"mv {dir}/obj_mask/{mask_name}.png {dir}/obj_mask/{new_mask_name}.png")
        
    if scan_id[:14] in test_scans:
        test_ids.append(new_image_name)
    elif scan_id[:14] in val_scans:
        val_ids.append(new_image_name)
    elif scan_id[:14] in train_scans:
        train_ids.append(new_image_name)

    return test_ids, train_ids, val_ids


def motion_real_statistics(data_dir, name_map_path, datasets, scan_split_dir):
    with open(name_map_path) as f:
        name_map = json.load(f)
    
    test_scan_list = f"{scan_split_dir}/test_scan.json"
    val_scan_list = f"{scan_split_dir}/valid_scan.json"
    train_scan_list = f"{scan_split_dir}/train_scan.json"

    TESTIDSPATH = f'{scan_split_dir}/test_scanids.json'
    VALIDSIDPATH = f'{scan_split_dir}/val_scanids.json'
    TRAINIDSPATH = f'{scan_split_dir}/train_scanids.json'

    test_ids_file = open(test_scan_list)
    test_scans = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(val_scan_list)
    val_scans = json.load(valid_ids_file)
    valid_ids_file.close()

    train_ids_file = open(train_scan_list)
    train_scans = json.load(train_ids_file)
    train_ids_file.close()

    test_ids = []
    val_ids = []
    train_ids = []

    current_object = {}
    dir = data_dir
    model_name = dir.split('/')[-1]
    current_object[model_name] = glob.glob(f"{dir}/rgb/*")

    # for image in current_object[model_name]:
    #     output = change_file_name(dir, name_map, test_ids, train_ids, val_ids, test_scans, val_scans, train_scans, image)
    #     import pdb
    #     pdb.set_trace()

    partial_process = partial(change_file_name, dir, name_map, test_ids, train_ids, val_ids, test_scans, val_scans, train_scans)
    results = process_map(partial_process, current_object[model_name], chunksize=1)

    # Merge results
    for result in results:
        test_ids += result[0]
        train_ids += result[1]
        val_ids += result[2]

    val_ids_file = open(VALIDSIDPATH, 'w')
    json.dump(val_ids, val_ids_file)
    val_ids_file.close()

    test_ids_file = open(TESTIDSPATH, 'w')
    json.dump(test_ids, test_ids_file)
    test_ids_file.close()

    train_ids_file = open(TRAINIDSPATH, 'w')
    json.dump(train_ids, train_ids_file)
    train_ids_file.close()


# split the data into train, validation, test sets
def copy_file(model_path, train_test, output_dir):
    file_path = model_path
    process_path = output_dir + train_test + '/' + \
        file_path.split('/')[-2] + '/' + file_path.split('/')[-1]
    os.system(f'cp {file_path} {process_path}')

def save_file(model_id, model_path, test_ids, valid_ids, train_ids, output_dir):
    if model_id in test_ids:
        copy_file(model_path, 'test', output_dir)
        # pool.apply_async(process, (model_path, 'test', output_dir))
    elif model_id in valid_ids:
        copy_file(model_path, 'valid', output_dir)
        # pool.apply_async(process, (model_path, 'valid', output_dir))
    elif model_id in train_ids:
        copy_file(model_path, 'train', output_dir)
        # pool.apply_async(process, (model_path, 'train', output_dir))


def split(anno_path, types, test_ids, valid_ids, train_ids, output_dir):
    model_id = anno_path.split('/')[-1].split('.')[0].split("_")[0]
    anno_file = open(anno_path)
    anno = json.load(anno_file)
    anno_file.close()

    for t in types:
            temp_path = anno_path.replace("annotation", t)
            tmp = temp_path.split(".")[0]
            paths = glob.glob(f"{tmp}.*") + glob.glob(f"{tmp}_*")
            for model_path in paths:
                save_file(model_id, model_path, test_ids, valid_ids, train_ids, output_dir)


# Function to convert data to h5 format
def convert_h5(dir, input_dir, output_dir):
    files = sorted(glob.glob(f'{input_dir}/{dir}/*.png'))
    num_files = len(files)

    with h5py.File(f'{output_dir}/{dir}.h5', "a") as h5file:
        if dir == 'depth':
            first_img = img = np.asarray(Image.open(files[0]), dtype=np.float32)[:, :, None]
        else:
            first_img = img = np.asarray(Image.open(files[0]).convert("RGB"))
        img_shape = first_img.shape
        img_dtype = first_img.dtype
        dataset_shape = (num_files,) + img_shape
        chunk_shape = (1,) + img_shape
        string_dtype = h5py.string_dtype(encoding="utf-8")
        # create image dataset tensor
        dset_images = h5file.create_dataset(
            f"{dir}_images",
            shape=dataset_shape,
            dtype=img_dtype,
            chunks=chunk_shape,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        # create image filenames dataset
        dset_filenames = h5file.create_dataset(
            f"{dir}_filenames", shape=(num_files,), dtype=string_dtype
        )
        # now fill the data
        for i in range(num_files):
            file = files[i]
            filepath = os.path.relpath(file, start=f'{input_dir}/{dir}')
            if dir == 'depth':
                img = np.asarray(Image.open(file), dtype=np.float32)[:, :, None]
            else:
                img = np.asarray(Image.open(file).convert("RGB"))
            dset_images[i] = img
            dset_filenames[i] = filepath
            # if i % 1000 == 0:
            #     print(f"{dir}: {i}/{num_files}")