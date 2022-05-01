import numpy as np
from lib_edgenet360.py_cuda import lib_edgenet360_setup, get_point_cloud_stanford
import cv2
from lib_edgenet360.preproc import fix_heigth_stanford, get_limits, adjust_ceil
from lib_edgenet360.file_utils import get_file_prefixes_from_path
import os
import json
import pandas as pd

in_path = './Data/stanford'
processed_path = './Data/stanford_processed'


PI = 3.14159265
DISP_SCALE = 2.0
DISP_OFFSET = -120.0

BASELINE=0.264
V_UNIT=0.02 #Not important here



def process(in_depth_map, in_rgb_file, out_depth_map, out_rgb_file, cam_pose, gt_file):

    gt_point_cloud = np.array(pd.read_pickle(gt_file), dtype=np.float32)

    # Read JSON data into the datastore variable
    with open(cam_pose, 'r') as jf:
        datastore = json.load(jf)

    z_rt, x_rt, y_rt = datastore["camera_original_rotation"]

    point_cloud, depth_image, bgr_image = get_point_cloud_stanford(in_depth_map, in_rgb_file=in_rgb_file, y_rt=y_rt)

    front_dist, back_dist = np.max(gt_point_cloud[:, 2]), np.min(gt_point_cloud[:, 2])
    right_dist, left_dist = np.max(gt_point_cloud[:, 0]), np.min(gt_point_cloud[:, 0])

    fpc = gt_point_cloud[gt_point_cloud[:, 3] == 1]
    ceil_height = -np.percentile(-fpc[:,1],.1)
    fpc = gt_point_cloud[gt_point_cloud[:, 3] == 2]
    floor_height = np.percentile(fpc[:,1],.1)

    depth_image = fix_heigth_stanford(point_cloud, depth_image, floor_height, ceil_height)

    depth_image[point_cloud[:,:,0]>(right_dist)]=65535
    depth_image[point_cloud[:,:,0]<(left_dist)]=65535
    depth_image[point_cloud[:,:,2]>(front_dist)]=65535
    depth_image[point_cloud[:,:,2]<(back_dist)]=65535

    cv2.imwrite(out_depth_map, depth_image)
    cv2.imwrite(out_rgb_file, bgr_image)



lib_edgenet360_setup(device=0, num_threads=1024, v_unit=V_UNIT, v_margin=0.24, f=518.8579, debug=0)

files = get_file_prefixes_from_path(processed_path, criteria="*.pkl")
total = len(files)
for i, file in enumerate(files):
    file_path, base_name = os.path.split(file)
    area = file_path[len(processed_path)+1:]
    camera = base_name[7:39]
    room = base_name[40:-32]

    print("%d/%d" % (i,total), total, area, camera , room)

    in_depth_map = os.path.join(in_path, area, 'pano/depth/camera_'+camera+'_'+room+'_frame_equirectangular_domain_depth.png')
    in_rgb_file = os.path.join(in_path, area, 'pano/rgb/camera_'+camera+'_'+room+'_frame_equirectangular_domain_rgb.png')
    gt_file = os.path.join(processed_path, area, 'camera_'+camera+'_'+room+'_frame_equirectangular_domain_gt.pkl')
    cam_pose = os.path.join(in_path, area, 'pano/pose/camera_'+camera+'_'+room+'_frame_equirectangular_domain_pose.json')

    out_depth_map = os.path.join(processed_path, area, 'camera_'+camera+'_'+room+'_frame_equirectangular_domain_sdepth.png')
    out_rgb_file = os.path.join(processed_path, area,  'camera_'+camera+'_'+room+'_frame_equirectangular_domain_srgb.png')

    process(in_depth_map, in_rgb_file, out_depth_map, out_rgb_file, cam_pose, gt_file)


