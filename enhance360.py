import argparse

import numpy as np
from lib_edgenet360.py_cuda import lib_edgenet360_setup, get_point_cloud, get_edge_point_cloud, \
                                   get_voxels, downsample_grid, get_ftsdf
from lib_edgenet360.file_utils import voxel_export, rgb_voxel_export
import cv2
import matplotlib.pyplot as plt
from lib_edgenet360.preproc import plane_estimate, find_limits_v2, complete_depth, find_limits_edge, \
    find_planes, ang_disparity, pt_disparity, fix_limits
import matplotlib.pyplot as plt
import os

PI = 3.14159265
DISP_SCALE = 2.0
DISP_OFFSET = -120.0

BASELINE=0.264
V_UNIT=0.02 #Not important here
DATA_PATH = './Data'

depth_file = './Data/Usability/shifted-disparity.png'
out_depth_file = './Data/Usability/new_shifted-disparity.png'
rgb_file = './Data/Usability/shifted_t.png'
out_prefix = 'Usability_enhanced'


def process(depth_file, rgb_file, out_depth_file, baseline):

    #cv2.namedWindow("Work", flags=cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Output", flags=cv2.WINDOW_NORMAL)

    print(depth_file)
    print(rgb_file)

    lib_edgenet360_setup(device=0, num_threads=1024, v_unit=V_UNIT, v_margin=0.24, f=518.8579, debug=0)

    point_cloud, depth_image = get_point_cloud(depth_file, baseline=baseline)
    #cv2.imshow("Output", depth_image)
    #cv2.waitKey(1)

    bgr_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)

    edges_image = cv2.Canny(bgr_image, 30, 70)

    kernel = np.ones((5,5),np.uint8)

    thin_edges = cv2.dilate(edges_image,kernel,iterations = 1)
    wide_edges = cv2.dilate(edges_image,kernel,iterations = 3)

    bilateral=cv2.bilateralFilter(bgr_image,3,75,75)

    new_depth_image, region_mask, edges_mask, inf_region_mask, close_region_mask = find_planes(point_cloud, bilateral, wide_edges, depth_image, thin_edges, baseline=baseline)

    cv2.imwrite(out_depth_file, new_depth_image)

    point_cloud, depth_image = get_point_cloud(out_depth_file, baseline=baseline)

    #DWRC
    #Baseline: 0.264m
    #Capture height: 1.45m
    #Room height: 2.328m
    #Room width: 4.279m
    #Room length: 5.613m

    #Usability
    #baseline: 0.264m
    #Capture height: 1.45m
    #Room width: 4.77m - 5.20m
    #Room length: 5.57m
    #Room height: 2.91m

    ceil_height, floor_height, front_dist, back_dist, right_dist, left_dist = find_limits_v2(point_cloud)
    #ceil_height, floor_height, front_dist, back_dist, right_dist, left_dist = find_limits_edge(edge_point_cloud)

    print("room height: %2.2f (%2.2f <> %2.2f)" % (ceil_height - floor_height, ceil_height, floor_height))
    print("room width:  %2.2f (%2.2f <> %2.2f)" % ( right_dist - left_dist, right_dist , left_dist))
    print("room length: %2.2f (%2.2f <> %2.2f)" % ( front_dist - back_dist, front_dist , back_dist))

    fixed_depth = fix_limits(point_cloud, depth_image,
                             ceil_height, floor_height, front_dist, back_dist, right_dist, left_dist,
                             baseline=baseline)
    #cv2.imshow("Output", fixed_depth)
    cv2.imwrite(out_depth_file, fixed_depth)
    #print("Finished! Press q!")

    #while True:
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break


def parse_arguments():
    global DATA_PATH

    print("\n360 depth maps enhancer\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",         help="360 dataset dir", type=str)
    parser.add_argument("depth_map",    help="360 depth map", type=str)
    parser.add_argument("rgb",          help="360 rgb", type=str)
    parser.add_argument("output",       help="output file prefix", type=str)
    parser.add_argument("--baseline",   help="Stereo 360 camera baseline. Default 0.264", type=float, default=0.264, required=False)
    parser.add_argument("--data_path",     help="Data path. Default %s"%DATA_PATH, type=str,
                                           default=DATA_PATH, required=False)

    args = parser.parse_args()
    DATA_PATH = args.data_path
    dataset = args.dataset

    depth_map = os.path.join(DATA_PATH, dataset, args.depth_map)
    rgb_file = os.path.join(DATA_PATH,dataset, args.rgb)
    output = os.path.join(DATA_PATH, dataset, args.output)
    baseline = args.baseline

    return depth_map, rgb_file, output, baseline

# Main Function
def Run():
    depth_map, rgb_file, output, baseline = parse_arguments()
    process(depth_map, rgb_file, output, baseline)


if __name__ == '__main__':
  Run()
