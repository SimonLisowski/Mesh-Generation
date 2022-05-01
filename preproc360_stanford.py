import argparse

import numpy as np
from lib_edgenet360.py_cuda import lib_edgenet360_setup, get_point_cloud_stanford
import cv2
from lib_edgenet360.preproc import fix_heigth_stanford, get_limits, adjust_ceil
import os
import json
import pandas as pd

PI = 3.14159265
DISP_SCALE = 2.0
DISP_OFFSET = -120.0

BASELINE=0.264
V_UNIT=0.02 #Not important here


#in_path = '/media/ad01345/Seagate Expansion Drive/stanford'
#out_path = '/media/ad01345/Seagate Expansion Drive/stanford_processed'

#in_path = './Data/stanford'
#out_path = './Data/stanford_processed'


#depth_file = './depth/camera_0e30c45ea0604ddeb7467fd384362503_office_7_frame_equirectangular_domain_depth.png'
#rgb_file   = './rgb/camera_0e30c45ea0604ddeb7467fd384362503_office_7_frame_equirectangular_domain_rgb.png'


def process(in_depth_map, in_rgb_file, out_depth_map, out_rgb_file, cam_pose, gt_file):

    print("Geting gt cloud from:", gt_file)
    gt_point_cloud = np.array(pd.read_pickle(gt_file), dtype=np.float32)


    # Read JSON data into the datastore variable
    with open(cam_pose, 'r') as jf:
        datastore = json.load(jf)

    z_loc, x_loc, y_loc = datastore["camera_location"]
    z_rt, x_rt, y_rt = datastore["camera_original_rotation"]

    print("Camera_position: (%2.2f - %2.2f - %2.2f)" % (x_loc, y_loc, z_loc))
    print("Camera_rotation: (%2.2f - %2.2f - %2.2f)" % (x_rt*180/PI , y_rt*180/PI, z_rt*180/PI))


    lib_edgenet360_setup(device=0, num_threads=1024, v_unit=V_UNIT, v_margin=0.24, f=518.8579, debug=0)

    point_cloud, depth_image, bgr_image = get_point_cloud_stanford(in_depth_map, in_rgb_file=in_rgb_file, y_rt=y_rt)

    front_dist, back_dist = np.max(point_cloud[:,:, 2]), np.min(point_cloud[:,:, 2])
    right_dist, left_dist = np.max(point_cloud[:,:, 0]), np.min(point_cloud[:,:, 0])
    floor_height, ceil_height = np.max(point_cloud[:,:, 1]), np.min(point_cloud[:,:, 1])

    print("INPUT Limits:")
    print("room height: %2.2f (%2.2f <> %2.2f)" % (ceil_height - floor_height, floor_height, ceil_height))
    print("room width:  %2.2f (%2.2f <> %2.2f)" % ( right_dist - left_dist , left_dist, right_dist))
    print("room length: %2.2f (%2.2f <> %2.2f)" % ( front_dist - back_dist , back_dist, front_dist))



    #floor_height, ceil_height = get_limits(point_cloud[:, :, 1])

    #ceil_height = adjust_ceil(point_cloud, ceil_height, .20)

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



    print("GT Limits:")
    print("room height: %2.2f (%2.2f <> %2.2f)" % (ceil_height - floor_height, floor_height, ceil_height))
    print("room width:  %2.2f (%2.2f <> %2.2f)" % ( right_dist - left_dist , left_dist, right_dist))
    print("room length: %2.2f (%2.2f <> %2.2f)" % ( front_dist - back_dist , back_dist, front_dist))

    cv2.imwrite(out_depth_map, depth_image)
    cv2.imwrite(out_rgb_file, bgr_image)

    point_cloud, depth_image = get_point_cloud_stanford(out_depth_map, in_rgb_file=None, y_rt=None)

    front_dist, back_dist = np.max(point_cloud[:,:, 2]), np.min(point_cloud[:,:, 2])
    right_dist, left_dist = np.max(point_cloud[:,:, 0]), np.min(point_cloud[:,:, 0])
    ceil_height, floor_height = np.max(point_cloud[:,:, 1]), np.min(point_cloud[:,:, 1])

    print("New INPUT Limits:")
    print("room height: %2.2f (%2.2f <> %2.2f)" % (ceil_height - floor_height, floor_height, ceil_height))
    print("room width:  %2.2f (%2.2f <> %2.2f)" % ( right_dist - left_dist , left_dist, right_dist))
    print("room length: %2.2f (%2.2f <> %2.2f)" % ( front_dist - back_dist , back_dist, front_dist))



def parse_arguments():

    print("\n360 depth maps enhancer\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("area",    help="Area", type=str)
    parser.add_argument("room",    help="Room", type=str)
    parser.add_argument("camera",  help="Camera", type=str)
    parser.add_argument("--in_path",   help="Stanford 2D-3D-Semantics dataset root", type=str, default=in_path, required=False)
    parser.add_argument("--out_path",   help="Output base path", type=str, default=out_path, required=False)

    args = parser.parse_args()
    
    in_depth_map = os.path.join(in_path, args.area, 'pano/depth/camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_depth.png')
    in_rgb_file = os.path.join(in_path, args.area, 'pano/rgb/camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_rgb.png')
    
    gt_file = os.path.join(out_path, args.area, 'camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_gt.pkl')
    cam_pose = os.path.join(in_path, args.area, 'pano/pose/camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_pose.json')

    out_depth_map = os.path.join(out_path, args.area, 'camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_sdepth.png')
    out_rgb_file = os.path.join(out_path, args.area,  'camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_srgb.png')

    

    return in_depth_map, in_rgb_file, out_depth_map, out_rgb_file, cam_pose, gt_file

# Main Function
def Run():
    ####################################################################################################################
### mine
    
    in_depth_map = './Data/stanford/area_3/pano/depth/camera_87d7995a0bc84ba49fee201a3e416828_conferenceRoom_1_frame_equirectangular_domain_depth.png' 
    in_rgb_file = './Data/stanford/area_3/pano/rgb/camera_87d7995a0bc84ba49fee201a3e416828_conferenceRoom_1_frame_equirectangular_domain_rgb.png'
    
    gt_file = './Data/stanford_processed/area_3/camera_87d7995a0bc84ba49fee201a3e416828_conferenceRoom_1_frame_equirectangular_domain_gt.pkl'
    cam_pose = './Data/stanford/area_3/pano/pose/camera_87d7995a0bc84ba49fee201a3e416828_conferenceRoom_1_frame_equirectangular_domain_pose.json'

    out_depth_map ='./Data/stanford_processed/area_3/out/camera_87d7995a0bc84ba49fee201a3e416828_conferenceRoom_1_frame_equirectangular_domain_sdepth.png'
    out_rgb_file = './Data/stanford_processed/area_3/out/camera_87d7995a0bc84ba49fee201a3e416828_conferenceRoom_1_frame_equirectangular_domain_srgb.png'
    
    #######################################################################################################################
    print("''''''''''''''1")
    print("Depth map: " + in_depth_map)
    #in_depth_map, in_rgb_file, out_depth_map, out_rgb_file, cam_pose, gt_file = parse_arguments()
    process(in_depth_map, in_rgb_file, out_depth_map, out_rgb_file, cam_pose, gt_file)


if __name__ == '__main__':
 Run()
