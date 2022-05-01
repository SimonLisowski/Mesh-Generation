import argparse
import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

processed_path = './Data/stanford_processed'


from lib_edgenet360.py_cuda import lib_edgenet360_setup, get_point_cloud_stanford, \
    get_voxels, downsample_grid, get_ftsdf_stanford, downsample_limits, get_gt, get_class_names
from lib_edgenet360.file_utils import voxel_export, get_file_prefixes_from_path
from lib_edgenet360.network import get_network_by_name
from lib_edgenet360.metrics import comp_iou, seg_iou, comp_iou_stanford, seg_iou_stanford
from lib_edgenet360.losses import weighted_categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import cv2

CV_PI = 3.141592
#f 518.8579

prediction_shape = (60,36,60)
probs_shape = (60,36,60,12)

#base_path = '/media/ad01345/Seagate Expansion Drive/stanford_processed'
processed_path = './Data/stanford_processed'


CSV_FILE = "results.csv"
CSV_COLS = ['area', 'room', 'camera', 'comp_iou', "ceil", "floor", "wall", "wind", "chair", "bed", "sofa", "table", "tvs", "furn", "objs", 'avg']
class_names = ["ceil", "floor", "wall", "wind", "chair", "bed", "sofa", "table", "tvs", "furn", "objs"]

def process(depth_file, rgb_file, gt_file, model):

    gt_point_cloud = np.array(pd.read_pickle(gt_file), dtype=np.float32)
    point_cloud, depth_image = get_point_cloud_stanford(depth_file)

    ceil_height, floor_height = np.max(point_cloud[:,:, 1])+0.02, np.min(point_cloud[:,:, 1]) - 0.04
    front_dist, back_dist = np.max(gt_point_cloud[:, 2])+0.04, np.min(gt_point_cloud[:, 2])-0.04
    right_dist, left_dist = np.max(gt_point_cloud[:, 0]+0.04), np.min(gt_point_cloud[:, 0])-0.04

    bgr_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    edges_image = cv2.Canny(bgr_image,100,200)

    xs, ys, zs = prediction_shape

    pred_full = np.zeros((xs*2,ys,xs*2,12), dtype=np.float32)
    surf_full = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
    flags_full = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)

    for ix in range(8):

        view = ix + 1

        print(view, end= " ")

        vox_grid, vox_grid_edges = get_voxels(point_cloud, depth_image.shape, edges_image,
                                              min_x=left_dist, max_x=right_dist,
                                              min_y=floor_height, max_y=ceil_height,
                                              min_z=back_dist, max_z=front_dist,
                                              vol_number=view)
        vox_grid_down = downsample_grid(vox_grid)


        vox_tsdf, vox_tsdf_edges, vox_limits = get_ftsdf_stanford(depth_image, vox_grid, vox_grid_edges,
                                                                  min_x=left_dist, max_x=right_dist,
                                                                  min_y=floor_height, max_y=ceil_height,
                                                                  min_z=back_dist, max_z=front_dist, vol_number=view)

        if network_type=='depth':
            x = vox_tsdf.reshape(1,240,144,240,1)
        elif network_type=='edges':
            x = [vox_tsdf.reshape(1,240,144,240,1),vox_tsdf_edges.reshape(1,240,144,240,1)]
        else:
            raise Exception('Invalid network tyoe: {}'.format(network_type))

        pred = np.argmax(model.predict(x=x), axis=-1)


        flags_down = downsample_limits(vox_limits)

        fpred =  pred.reshape((zs,ys,xs,12)) * np.repeat(flags_down,12).reshape((zs,ys,xs,12))

        if view==1:
            pred_full[ zs:, :, xs//2:-xs//2] += fpred
            surf_full[ zs:, :, xs//2:-xs//2] |= vox_grid_down
            flags_full[ zs:, :, xs//2:-xs//2] |= flags_down

        elif view==2:
            pred_full[zs:, :, xs:] += fpred
            surf_full[zs:, :, xs:] |= vox_grid_down
            flags_full[zs:, :, xs:] |= flags_down

        elif view==3:
            pred_full[zs//2:-zs//2, :, xs:] += fpred
            surf_full[zs//2:-zs//2, :, xs:] |= vox_grid_down
            flags_full[zs//2:-zs//2, :, xs:] |= flags_down

        elif view == 4:
            pred_full[:zs, :, xs:] += np.flip(np.swapaxes(fpred,0,2),axis=0)
            surf_full[:zs, :, xs:] |= np.flip(np.swapaxes(vox_grid_down,0,2),axis=0)
            flags_full[:zs, :, xs:] |= np.flip(np.swapaxes(flags_down,0,2),axis=0)

        elif view==5:
            pred_full[:zs, :, xs//2:-xs//2] += np.flip(fpred,axis=[0,2])
            surf_full[:zs, :, xs//2:-xs//2] |= np.flip(vox_grid_down,axis=[0,2])
            flags_full[:zs, :, xs//2:-xs//2] |= np.flip(flags_down,axis=[0,2])
        elif view==6:
            pred_full[:zs, :, :xs] += np.flip(fpred,axis=[0,2])
            surf_full[:zs, :, :xs] |= np.flip(vox_grid_down,axis=[0,2])
            flags_full[:zs, :, :xs] |= np.flip(flags_down,axis=[0,2])

        elif view==7:
            pred_full[zs//2:-zs//2, :,:xs ] += np.flip(fpred,axis=[0,2])
            surf_full[zs//2:-zs//2, :,:xs ] |= np.flip(vox_grid_down,axis=[0,2])
            flags_full[zs//2:-zs//2, :,:xs ] |= np.flip(flags_down,axis=[0,2])

        elif view == 8:
            pred_full[zs:, :, :xs] += np.flip(fpred,axis=2)
            surf_full[zs:, :, :xs] |= np.flip(vox_grid_down,axis=2)
            flags_full[zs:, :, :xs] |= np.flip(flags_down,axis=2)

    gt_grid = get_gt(gt_point_cloud,
                     min_x=left_dist, max_x=right_dist,
                     min_y=floor_height, max_y=ceil_height,
                     min_z=back_dist, max_z=front_dist)

    gt_grid = gt_grid * flags_full

    y_pred = np.argmax(pred_full, axis=-1)
    # fill camera position
    y_pred[zs-4:zs+4,0,xs-4:xs+4] = 2

    # class mappings
    y_pred[y_pred == 6] = 8  # bed -> table
    y_pred[y_pred == 9] = 11  # tv -> objects
    pred_full = to_categorical(y_pred, num_classes=12)



    #evaluation

    y_true =  to_categorical(gt_grid, num_classes=12)

    seg_inter = np.zeros((11,),dtype=float)
    seg_union = np.zeros((11,),dtype=float)

    comp_inter, comp_union, tp, fp, fn = comp_iou_stanford(y_true, pred_full, surf_full)

    for cl in range(0, 11):
        seg_inter[cl], seg_union[cl] = seg_iou_stanford(y_true, pred_full, cl + 1)

    return comp_inter, comp_union, tp, fp, fn, seg_inter, seg_union



lib_edgenet360_setup(device=0, num_threads=1024, v_unit=0.021, v_margin=0.24, f=518.8579, debug=0)

network="EdgeNet"
print("\nLoading %s..." % network)

model, network_type = get_network_by_name(network)
model.compile(optimizer=SGD(lr=0.01, decay=0.005, momentum=0.9),
              loss=weighted_categorical_crossentropy
              , metrics=[comp_iou, seg_iou]
              )

weight_file = {'USSCNet': 'R_UNET_LR0.01_DC0.0005_621-0.69-0.54.hdf5',
               'EdgeNet': 'R_UNET_E_LR0.01_DC0.0005_4535-0.77-0.55.hdf5'
               # 'EdgeNet': 'R_UNET_E_LR0.01_DC0.0005_4318-0.67-0.54.hdf5'
               }

model.load_weights(os.path.join('./weights', weight_file[network]))

files = get_file_prefixes_from_path(processed_path, criteria="*.pkl")
total = len(files)

sum_seg_inter = np.zeros((11,), dtype=float)
sum_seg_union = np.zeros((11,), dtype=float)
sum_comp_inter, sum_comp_union, sum_tp, sum_fp, sum_fn = 0, 0, 0, 0, 0

room_types = ['office', 'conferenceRoom', 'pantry', 'copyRoom', 'storage' ]

print("      ####/#### ",end="")
for class_name in ['prec.','recall', 'CIoU'] + class_names:
    print((class_name+'      ')[:8], end="")
print("avg       # # # # # # # #")

for i, file in enumerate(files):
    file_path, base_name = os.path.split(file)
    area = file_path[len(processed_path)+1:]
    camera = base_name[7:39]
    room = base_name[40:-32]

    if room.split("_")[0] not in room_types:
        continue

    depth_map = os.path.join(processed_path, area, 'camera_'+camera+'_'+room+'_frame_equirectangular_domain_sdepth.png')
    rgb_file = os.path.join(processed_path, area, 'camera_'+camera+'_'+room+'_frame_equirectangular_domain_srgb.png')
    gt_file = os.path.join(processed_path, area, 'camera_'+camera+'_'+room+'_frame_equirectangular_domain_gt.pkl')

    comp_inter, comp_union, tp, fp, fn, seg_inter, seg_union = process(depth_map, rgb_file, gt_file, model)

    print(end="\r")
    sum_seg_inter += seg_inter
    sum_seg_union += seg_union
    sum_comp_inter += comp_inter
    sum_comp_union += comp_union
    sum_tp += tp
    sum_fp += fp
    sum_fn += fn

    comp_iou = 100 * (comp_inter + 0.0000000001)/(comp_union + 0.0000000002)
    seg_iou = 100 * (seg_inter + 0.0000000001)/(seg_union + 0.0000000002)

    resdf = pd.DataFrame({'area': [area], 'room':[room], 'camera': [camera],
                          'comp_iou': [round(comp_iou,2)]})
    avg = 0
    qt = 0
    for cl in range(0, 11):
        resdf[class_names[cl]] = [round(seg_iou[cl],2)]
        if cl in [5, 8]:
            continue
        avg += seg_iou[cl]
        qt += 1
    avg /= qt

    resdf['avg']=[round(avg,2)]

    if os.path.isfile(CSV_FILE):
       with open(CSV_FILE, 'a') as f:
           resdf.to_csv(f, header=False, index=False, columns=CSV_COLS)
    else:
           resdf.to_csv(CSV_FILE,header=True, index=False, mode='w', columns=CSV_COLS)


    if (sum_tp + sum_fp) > 0:
        precision = sum_tp / (sum_tp + sum_fp)
    else:
        precision = 0

    if (sum_tp + sum_fn) > 0:
        recall = sum_tp / (sum_tp + sum_fn)
    else:
        recall = 0

    comp_iou = 100 * (sum_comp_inter + 0.0000000001)/(sum_comp_union + 0.0000000002)
    seg_iou = 100 * (sum_seg_inter + 0.0000000001)/(sum_seg_union + 0.0000000002)

    print("      %4d/%4d %4.1f    %4.1f    %4.1f    " % (i,total, precision, recall, comp_iou), end="")

    avg = 0
    qt = 0
    for cl in range(0, 11):
        print("%4.1f   " % seg_iou[cl], end = " ")
        if cl in [5, 8]:
            continue
        avg += seg_iou[cl]
        qt += 1
    avg /= qt

    print("%4.1f    " % avg, end="  ")

print("\n")
