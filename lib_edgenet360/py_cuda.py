import ctypes
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from enum import Enum
class InputType(Enum):
   DEPTH_ONLY = 1
   DEPTH_COLOR = 2
   DEPTH_EDGES = 3

def get_segmentation_class_map():
    return np.array([0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11,
                                   11, 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11], dtype=np.int32)
def get_class_names():
    return ["ceil.", "floor", "wall ", "wind.", "chair", "bed  ", "sofa ", "table", "tvs  ", "furn.", "objs."]


#nvcc --ptxas-options=-v --compiler-options '-fPIC' -o lib_edgenet360.so --shared -std=c++11 lib_edgenet360.cu

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'src/lib_edgenet360.so'))


_lib.setup.argtypes = (ctypes.c_int,
              ctypes.c_int,
              ctypes.c_float,
              ctypes.c_float,
              ctypes.c_float,
              ctypes.c_float,
              ctypes.c_float,
              ctypes.c_int,
              ctypes.c_int,
              ctypes.c_int,
              ctypes.c_int)

voxel_shape = (240,144,240)

def lib_edgenet360_setup(device=0, num_threads=1024, v_unit=0.02, v_margin=0.24,
                         f=518.8, sensor_w=640, sensor_h=480,
                         vox_shape=None,
                         debug=0):

    global _lib, voxel_shape

    if vox_shape is not None:
        voxel_shape = vox_shape



    _lib.setup(ctypes.c_int(device),
                  ctypes.c_int(num_threads),
                  ctypes.c_float(v_unit),
                  ctypes.c_float(v_margin),
                  ctypes.c_float(f),
                  ctypes.c_float(sensor_w),
                  ctypes.c_float(sensor_h),
                  ctypes.c_int(voxel_shape[0]),
                  ctypes.c_int(voxel_shape[1]),
                  ctypes.c_int(voxel_shape[2]),
                  ctypes.c_int(debug)
               )


_lib.get_point_cloud.argtypes = (ctypes.c_float,
                          ctypes.c_void_p,
                          ctypes.c_void_p,
                          ctypes.c_int,
                          ctypes.c_int,
                         )

def get_point_cloud(depth_file, baseline):
    global _lib

    depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

    height, width = depth_image.shape
    #depth_image[:, :] = 130

    '''
    #for i in range(width):
    #  depth_image[:, i] = i%140

    depth_image[0:height//2, 672:672+200] = 90 #grid 0k
    depth_image[height//2:height, 672-200:672] = 60 #grid 0k

    '''
    num_pixels = height * width

    point_cloud = np.zeros((height, width, 6), dtype=np.float32)

    plt.imshow(depth_image)

    _lib.get_point_cloud(ctypes.c_float(baseline),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      point_cloud.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(width),
                      ctypes.c_int(height)
                      )
    return point_cloud, depth_image

def get_point_cloud_stanford(in_depth_map, in_rgb_file=None, y_rt=None):
    global _lib

    out_bgr_image=None

    PI = 3.14159265

    in_depth_image = cv2.imread(in_depth_map, cv2.IMREAD_ANYDEPTH)
    out_depth_image = in_depth_image.copy()


    height, width = in_depth_image.shape

    if not y_rt is None:

        shift_point = int((y_rt * width) / (2*PI))

        out_depth_image[:,-shift_point:] = in_depth_image[:,:shift_point]
        out_depth_image[:,:-shift_point] = in_depth_image[:,shift_point:]

        if not in_rgb_file is None:
            in_bgr_image = cv2.imread(in_rgb_file, cv2.IMREAD_COLOR)
            out_bgr_image = in_bgr_image.copy()
            out_bgr_image[:,-shift_point:] = in_bgr_image[:,:shift_point]
            out_bgr_image[:,:-shift_point] = in_bgr_image[:,shift_point:]

    point_cloud = np.zeros((height, width, 6), dtype=np.float32)

    _lib.get_point_cloud_stanford(out_depth_image.ctypes.data_as(ctypes.c_void_p),
                      point_cloud.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(width),
                      ctypes.c_int(height)
                      )

    if not out_bgr_image is None:
        return point_cloud, out_depth_image, out_bgr_image
    else:
        return point_cloud, out_depth_image

def get_edge_point_cloud(rgb_file, depth_file, baseline):
    global _lib

    depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    depth_image[:250,:] = 0
    depth_image[-250:,:] = 0

    rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)

    #edges_image = cv2.Canny(rgb_image, 100, 200)
    edges_image = cv2.Canny(rgb_image, 33, 80)
    height, width = depth_image.shape
    edges_image[:250] = 0
    edges_image[-250:] = 0

    thin_edges =  edges_image.copy()

    linek = np.zeros((7, 7), dtype=np.uint8)
    linek[...,3] = 1
    x = cv2.morphologyEx(edges_image, cv2.MORPH_OPEN, linek, iterations=1)

    #f, axarr = plt.subplots(2,1)
    #axarr[0].imshow(edges_image)
    #axarr[1].imshow(edges_image-x)

    edges_image -= x


    for i in range(4):
        edges_image[i+1:height,:] = edges_image[i+1:height,:] | edges_image[0:height-i-1,:]
        edges_image[0:height-i-1,:] = edges_image[0:height-i-1,:] | edges_image[i+1:height,:]
        edges_image[:,i+1:width] = edges_image[:,i+1:width] | edges_image[:,0:width-i-1]
        edges_image[:,0:width-i-1] = edges_image[:,0:width-i-1] | edges_image[:,i+1:width]

    for i in range(1):
        thin_edges[i+1:height,:] = thin_edges[i+1:height,:] | thin_edges[0:height-i-1,:]
        thin_edges[0:height-i-1,:] = thin_edges[0:height-i-1,:] | thin_edges[i+1:height,:]
        thin_edges[:,i+1:width] = thin_edges[:,i+1:width] | thin_edges[:,0:width-i-1]
        thin_edges[:,0:width-i-1] = thin_edges[:,0:width-i-1] | thin_edges[:,i+1:width]



    combined = depth_image * (edges_image // 255)

    #depth_image[:, :] = 130

    '''
    #for i in range(width):
    #  depth_image[:, i] = i%140

    depth_image[0:height//2, 672:672+200] = 90 #grid 0k
    depth_image[height//2:height, 672-200:672] = 60 #grid 0k

    '''
    num_pixels = height * width

    point_cloud = np.zeros((height, width, 6), dtype=np.float32)

    _lib.get_point_cloud(ctypes.c_float(baseline),
                      combined.ctypes.data_as(ctypes.c_void_p),
                      point_cloud.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(width),
                      ctypes.c_int(height)
                      )
    return point_cloud, rgb_image, depth_image, edges_image, combined, thin_edges


_lib.get_voxels.argtypes = (ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_void_p,
                            ctypes.c_int,
                            ctypes.c_void_p,
                            ctypes.c_void_p
                            )

def get_voxels(point_cloud, point_cloud_shape, edges_image, min_x, max_x, min_y, max_y, min_z, max_z,
               vol_number=1):
    global _lib, voxel_shape

    boundaries = np.array([min_x, max_x, min_y, max_y, min_z, max_z], dtype=np.float32)
    vox_grid = np.zeros(voxel_shape,dtype=np.uint8)
    vox_grid_edges = np.zeros(voxel_shape,dtype=np.uint8)

    _lib.get_voxels(point_cloud.ctypes.data_as(ctypes.c_void_p),
                    edges_image.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(point_cloud_shape[1]),
                    ctypes.c_int(point_cloud_shape[0]),
                    boundaries.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(vol_number),
                    vox_grid.ctypes.data_as(ctypes.c_void_p),
                    vox_grid_edges.ctypes.data_as(ctypes.c_void_p)
    )
    return vox_grid, vox_grid_edges

_lib.get_gt.argtypes = (ctypes.c_void_p,
                            ctypes.c_int,
                            ctypes.c_void_p,
                            ctypes.c_void_p
                         )

def get_gt(point_cloud, min_x, max_x, min_y, max_y, min_z, max_z):
    global _lib, voxel_shape

    boundaries = np.array([min_x, max_x, min_y, max_y, min_z, max_z], dtype=np.float32)


    gt_grid = np.zeros((voxel_shape[0]//2, voxel_shape[1]//4, voxel_shape[2]//2),dtype=np.uint8)

    _lib.get_gt(point_cloud.flatten().ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(point_cloud.shape[0]),
                    boundaries.ctypes.data_as(ctypes.c_void_p),
                    gt_grid.ctypes.data_as(ctypes.c_void_p)
                      )
    return gt_grid


_lib.downsample_grid.argtypes = (ctypes.c_void_p,
                            ctypes.c_void_p
                            )

def downsample_grid(in_vox_grid):
    global _lib, voxel_shape

    out_vox_grid = np.zeros((voxel_shape[0]//4,voxel_shape[1]//4,voxel_shape[2]//4 ),dtype=np.uint8)

    _lib.downsample_grid(in_vox_grid.ctypes.data_as(ctypes.c_void_p),
                        out_vox_grid.ctypes.data_as(ctypes.c_void_p)
                      )
    return out_vox_grid

_lib.FTSDFDepth.argtypes = (ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_float,
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_void_p,
                            ctypes.c_int
                         )

_lib.downsample_limits.argtypes = (ctypes.c_void_p,
                            ctypes.c_void_p
                            )

def downsample_limits(in_vox_grid):
    global _lib, voxel_shape

    out_vox_grid = np.zeros((voxel_shape[0]//4,voxel_shape[1]//4,voxel_shape[2]//4 ),dtype=np.uint8)

    _lib.downsample_limits(in_vox_grid.ctypes.data_as(ctypes.c_void_p),
                        out_vox_grid.ctypes.data_as(ctypes.c_void_p)
                      )
    return out_vox_grid

_lib.FTSDFDepth.argtypes = (ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_float,
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_void_p,
                            ctypes.c_int
                         )


def get_ftsdf(depth_image, vox_grid, vox_grid_edges, min_x, max_x, min_y, max_y, min_z, max_z,
               baseline, vol_number=1):
    global _lib, voxel_shape

    boundaries = np.array([min_x, max_x, min_y, max_y, min_z, max_z], dtype=np.float32)
    height, width = depth_image.shape
    vox_tsdf = np.zeros(voxel_shape,dtype=np.float32)
    vox_tsdf_edges = np.zeros(voxel_shape,dtype=np.float32)
    vox_limits = np.zeros(voxel_shape,dtype=np.uint8)

    _lib.FTSDFDepth(depth_image.ctypes.data_as(ctypes.c_void_p),
                   vox_grid.ctypes.data_as(ctypes.c_void_p),
                   vox_grid_edges.ctypes.data_as(ctypes.c_void_p),
                   vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                   vox_tsdf_edges.ctypes.data_as(ctypes.c_void_p),
                   vox_limits.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_float(baseline),
                   ctypes.c_int(width),
                   ctypes.c_int(height),
                   boundaries.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(vol_number)
                  )
    return vox_tsdf, vox_tsdf_edges, vox_limits


_lib.FTSDFDepth_stanford.argtypes = (ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_void_p,
                            ctypes.c_int
                         )


def get_ftsdf_stanford(depth_image, vox_grid, vox_grid_edges, min_x, max_x, min_y, max_y, min_z, max_z, vol_number=1):
    global _lib, voxel_shape

    boundaries = np.array([min_x, max_x, min_y, max_y, min_z, max_z], dtype=np.float32)
    height, width = depth_image.shape
    vox_tsdf = np.zeros(voxel_shape,dtype=np.float32)
    vox_tsdf_edges = np.zeros(voxel_shape,dtype=np.float32)
    vox_limits = np.zeros(voxel_shape,dtype=np.uint8)

    _lib.FTSDFDepth_stanford(depth_image.ctypes.data_as(ctypes.c_void_p),
                   vox_grid.ctypes.data_as(ctypes.c_void_p),
                   vox_grid_edges.ctypes.data_as(ctypes.c_void_p),
                   vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                   vox_tsdf_edges.ctypes.data_as(ctypes.c_void_p),
                   vox_limits.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(width),
                   ctypes.c_int(height),
                   boundaries.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(vol_number)
                  )
    return vox_tsdf, vox_tsdf_edges, vox_limits




'''
_lib.get_rgb_grid.argtypes = (ctypes.c_float,
                          ctypes.c_void_p,
                          ctypes.c_void_p,
                          ctypes.c_void_p,
                          ctypes.c_void_p,
                         )

def get_rgb_voxel_grid(depth_file, rgb_file, patch_number, patch_width, patch_offset, baseline, voxel_shape):
    global _lib

    num_voxels = voxel_shape[0] * voxel_shape[1] * voxel_shape[2]
    down_shape = (voxel_shape[0]//4, voxel_shape[1]//4, voxel_shape[2]//4, 3)

    vox_size = np.array([voxel_shape[0], voxel_shape[1], voxel_shape[2]], dtype=np.int32)
    vox_grid = np.zeros(down_shape,dtype=np.uint8)

    depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    rgb_image = cv2.imread(rgb_file)
    rgb_image =cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)

    print("depth shape",depth_image.shape)

    rows, cols = depth_image.shape

    width = int(cols * patch_width)
    offset = int((cols * patch_offset) * patch_number)

    depth_image = depth_image[:,offset:(offset+width)].copy()
    rgb_image = rgb_image[:,offset:(offset+width)].copy()


    depth_image[:,:] = 0
    rgb_image[:,:] = 0
    x_offset = 3 * rgb_image.shape[0]//8
    y_offset = 3 * rgb_image.shape[1]//8
    colors = [[200,50, 0],[200,100, 0], [200,150,0],
              [0,50, 200],[0,100, 200], [0,150,200],
              [0,50, 0],[0,100, 0], [0,150,0],]
    x_ranges = [[x_offset + 0, x_offset + rgb_image.shape[0]//12],
                [x_offset + rgb_image.shape[0]//12, x_offset + 2 *rgb_image.shape[0]//12],
                [x_offset +  2 *rgb_image.shape[0]//12, x_offset + 3 *rgb_image.shape[0]//12]]
    y_ranges = [[y_offset + 0,y_offset + rgb_image.shape[1]//12],
                [y_offset + rgb_image.shape[1]//12, y_offset + 2 *rgb_image.shape[1]//12],
                [y_offset + 2 *rgb_image.shape[1]//12, y_offset + 3 *rgb_image.shape[1]//12]]
    colors = [[200,50, 0],[200,100, 0], [200,150,0],
              [0,50, 200],[0,100, 200], [0,150,200],
              [0,50, 0],[0,100, 0], [0,150,0],]

    x_ranges = [[0, rgb_image.shape[0]//3],
                [rgb_image.shape[0]//3, 2 *rgb_image.shape[0]//3],
                [2 *rgb_image.shape[0]//3, rgb_image.shape[0]]]
    y_ranges = [[0,rgb_image.shape[1]//3],
                [rgb_image.shape[1]//3, 2 *rgb_image.shape[1]//3],
                [2 *rgb_image.shape[1]//3, rgb_image.shape[1]]]

    for x in range(3):
        for y in range(3):
            c = colors[x*3+y]
            rgb_image[x_ranges[x][0]:x_ranges[x][1], y_ranges[y][0]:y_ranges[y][1]] = c
            depth_image[x_ranges[x][0]:x_ranges[x][1], y_ranges[y][0]:y_ranges[y][1]] = 100

    #depth_image[int(rows/2 -200):int(rows/2 +200) , int(width/2 - 200):int(width/2 +200)] = 100
    #depth_image[:,:] = 100
    #depth_image[0:100 , 0:20] = 100

    #plt.imshow(depth_image)
    print(np.max(depth_image))
    plt.imshow(rgb_image)

    _lib.get_rgb_grid(ctypes.c_float(baseline),
                      vox_size.ctypes.data_as(ctypes.c_void_p),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      rgb_image.ctypes.data_as(ctypes.c_void_p),
                      vox_grid.ctypes.data_as(ctypes.c_void_p)
                      )
    return vox_grid






_lib.ProcessColor.argtypes = (ctypes.c_char_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p
                              )


_lib.ProcessEdges.argtypes = (ctypes.c_char_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p
                              )


def process_color(file_prefix, voxel_shape, down_scale = 4):
    global _lib

    vox_origin = np.ones(3,dtype=np.float32)
    cam_pose = np.ones(16,dtype=np.float32)
    num_voxels = voxel_shape[0] * voxel_shape[1] * voxel_shape[2]
    vox_size = np.array([voxel_shape[0], voxel_shape[1], voxel_shape[2]], dtype=np.int32)
    segmentation_class_map = get_segmentation_class_map()
    segmentation_label = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.int32)

    vox_weights = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)
    vox_vol = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)

    depth_image = cv2.imread(file_prefix+'_depth.png', cv2.IMREAD_ANYDEPTH)
    rgb_image = cv2.imread(file_prefix+'_color.jpg', cv2.IMREAD_COLOR)

    vox_tsdf = np.zeros(num_voxels, dtype=np.float32)
    vox_rgb = np.zeros(3*num_voxels, dtype=np.float32)


    _lib.ProcessColor(ctypes.c_char_p(bytes(file_prefix+'.bin','utf-8')),
                      cam_pose.ctypes.data_as(ctypes.c_void_p),
                      vox_size.ctypes.data_as(ctypes.c_void_p),
                      vox_origin.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(down_scale),
                      segmentation_class_map.ctypes.data_as(ctypes.c_void_p),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      rgb_image.ctypes.data_as(ctypes.c_void_p),
                      vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                      vox_rgb.ctypes.data_as(ctypes.c_void_p),
                      vox_weights.ctypes.data_as(ctypes.c_void_p),
                      vox_vol.ctypes.data_as(ctypes.c_void_p),
                      segmentation_label.ctypes.data_as(ctypes.c_void_p)
                      )

    return vox_tsdf, vox_rgb, segmentation_label, vox_weights, vox_vol

def process_edges(file_prefix, voxel_shape, down_scale = 4):
    global _lib

    vox_origin = np.ones(3,dtype=np.float32)
    cam_pose = np.ones(16,dtype=np.float32)
    num_voxels = voxel_shape[0] * voxel_shape[1] * voxel_shape[2]
    vox_size = np.array([voxel_shape[0], voxel_shape[1], voxel_shape[2]], dtype=np.int32)
    segmentation_class_map = get_segmentation_class_map()
    segmentation_label = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.int32)

    vox_weights = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)
    vox_vol = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)

    depth_image = cv2.imread(file_prefix+'_depth.png', cv2.IMREAD_ANYDEPTH)
    #depth_image = cv2.imread(file_prefix+'.png', cv2.IMREAD_ANYDEPTH)
    rgb_image = cv2.imread(file_prefix+'_color.jpg', cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)
    edges_image = cv2.Canny(rgb_image,100,200)

    vox_tsdf = np.zeros(num_voxels, dtype=np.float32)
    tsdf_edges = np.zeros(num_voxels, dtype=np.float32)
    vox_edges = np.zeros(num_voxels, dtype=np.float32)


    _lib.ProcessEdges(ctypes.c_char_p(bytes(file_prefix+'.bin','utf-8')),
                      cam_pose.ctypes.data_as(ctypes.c_void_p),
                      vox_size.ctypes.data_as(ctypes.c_void_p),
                      vox_origin.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(down_scale),
                      segmentation_class_map.ctypes.data_as(ctypes.c_void_p),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      edges_image.ctypes.data_as(ctypes.c_void_p),
                      vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                      vox_edges.ctypes.data_as(ctypes.c_void_p),
                      tsdf_edges.ctypes.data_as(ctypes.c_void_p),
                      vox_weights.ctypes.data_as(ctypes.c_void_p),
                      vox_vol.ctypes.data_as(ctypes.c_void_p),
                      segmentation_label.ctypes.data_as(ctypes.c_void_p)
                      )

    return vox_tsdf, vox_edges, tsdf_edges, segmentation_label, vox_weights, vox_vol



def process(file_prefix, voxel_shape, down_scale = 4, input_type=InputType.DEPTH_COLOR):
    if input_type == InputType.DEPTH_COLOR:
        return process_color(file_prefix, voxel_shape, down_scale=down_scale)
    elif input_type == InputType.DEPTH_EDGES:
        return process_edges(file_prefix, voxel_shape, down_scale=down_scale)
    elif input_type == InputType.DEPTH_ONLY:
        print("input type DEPTH ONLY not implemented yet")
        exit(-1)
'''