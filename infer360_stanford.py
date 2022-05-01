import argparse
import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

CV_PI = 3.141592
#f 518.8579

prediction_shape = (60,36,60)
probs_shape = (60,36,60,12)

base_path = './Data/stanford_processed'
output_path = './Output'

########################################################################################################## start
WEIGHTS_PATH = './weights'
BASELINE = 0.264
V_UNIT = 0.02
NETWORK = 'EdgeNet'
FILTER = True
SMOOTHING = True
REMOVE_INTERNAL = False
MIN_VOXELS = 15
TRIANGULAR_FACES = False
FILL_LIMITS = True
INNER_FACES = False
INCLUDE_TOP = False
############################################################################################################ end



def process(depth_file, rgb_file, gt_file, out_prefix, v_unit, network):
    from lib_edgenet360.py_cuda import lib_edgenet360_setup, get_point_cloud_stanford, \
         get_voxels, downsample_grid, get_ftsdf_stanford, downsample_limits, get_gt, get_class_names
    from lib_edgenet360.file_utils import voxel_export
    from lib_edgenet360.network import get_network_by_name
    from lib_edgenet360.metrics import comp_iou, seg_iou, comp_iou_stanford, seg_iou_stanford
    from lib_edgenet360.losses import weighted_categorical_crossentropy
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.utils import to_categorical
    import cv2
    
####################################################################################################  start
    import numpy as np
    from lib_edgenet360.py_cuda import lib_edgenet360_setup, get_point_cloud, \
         get_voxels, downsample_grid, get_ftsdf, downsample_limits
    from lib_edgenet360.file_utils import obj_export
    from lib_edgenet360.post_process import voxel_filter, voxel_fill, fill_limits_vox, instance_remover,\
                                            remove_internal_voxels_v2
    ############################################################################################## end

    lib_edgenet360_setup(device=0, num_threads=1024, v_unit=v_unit, v_margin=0.24, f=518.8579, debug=0)

    print("Geting gt cloud from:", gt_file)
    gt_point_cloud = np.array(pd.read_pickle(gt_file), dtype=np.float32)


    print("Geting point cloud from:", depth_file)
    point_cloud, depth_image = get_point_cloud_stanford(depth_file)

    ceil_height, floor_height = np.max(point_cloud[:,:, 1])+0.02, np.min(point_cloud[:,:, 1]) - 0.04
    front_dist, back_dist = np.max(gt_point_cloud[:, 2])+0.04, np.min(gt_point_cloud[:, 2])-0.04
    right_dist, left_dist = np.max(gt_point_cloud[:, 0]+0.04), np.min(gt_point_cloud[:, 0])-0.04

    print("room height: %2.2f (%2.2f <> %2.2f)" % (ceil_height - floor_height, floor_height, ceil_height))
    print("room width:  %2.2f (%2.2f <> %2.2f)" % ( right_dist - left_dist , left_dist, right_dist))
    print("room length: %2.2f (%2.2f <> %2.2f)" % ( front_dist - back_dist , back_dist, front_dist))
####################################################################################################  start
    camx, camy, camz = -left_dist, -floor_height, -back_dist
####################################################################################################  end
    bgr_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    edges_image = cv2.Canny(bgr_image,100,200)

    print("\nLoading %s..." % network)
    model, network_type = get_network_by_name(network)
    model.compile(optimizer=SGD(lr=0.01, decay=0.005,  momentum=0.9),
                  loss=weighted_categorical_crossentropy
                  ,metrics=[comp_iou, seg_iou]
                  )

    weight_file = {'USSCNet': 'R_UNET_LR0.01_DC0.0005_621-0.69-0.54.hdf5',
                   'EdgeNet': 'R_UNET_E_LR0.01_DC0.0005_4535-0.77-0.55.hdf5'
                   #'EdgeNet': 'R_UNET_E_LR0.01_DC0.0005_4318-0.67-0.54.hdf5'
    }

    # model_name = os.path.join('./weights','SSCNET_E_LR0.01_DC0.0005_3524-0.69-0.70.hdf5')
    # model_name = os.path.join('./weights','SSCNET_LR0.01_DC0.0005_3926-0.68-0.70.hdf5')
    # model_name = os.path.join('./weights','R_UNET_LR0.01_DC0.0005_621-0.69-0.54.hdf5')

    model.load_weights(os.path.join('./weights',weight_file[network]))

    xs, ys, zs = prediction_shape

    vx_min = int(max(0, xs + left_dist//(v_unit*4)))
    vx_max = int(min(xs*2-1, xs + right_dist//(v_unit*4)))
    vz_min = int(max(0, zs + back_dist//(v_unit*4)))
    vz_max = int(min(zs*2-1, zs + front_dist//(v_unit*4)))



    pred_full = np.zeros((xs*2,ys,xs*2,12), dtype=np.float32)
    surf_full = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
    flags_full = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)

    for ix in range(8):

        view = ix + 1

        print("Processing view", view, end= "\r")

        vox_grid, vox_grid_edges = get_voxels(point_cloud, depth_image.shape, edges_image,
                                              min_x=left_dist, max_x=right_dist,
                                              min_y=floor_height, max_y=ceil_height,
                                              min_z=back_dist, max_z=front_dist,
                                              vol_number=view)

        vox_grid_down = downsample_grid(vox_grid)
        vox_grid_edges_down = downsample_grid(vox_grid_edges)

        vox_grid_down = vox_grid_down

        out_file = out_prefix+'_view'+str(view)+'_surface.bin'
        voxel_export(out_file, vox_grid_down, vox_grid_down.shape)

        out_file =out_prefix+'_view'+str(view)+'_edges.bin'
        voxel_export(out_file, vox_grid_edges_down, vox_grid_down.shape)

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

        pred = model.predict(x=x)

        flags_down = downsample_limits(vox_limits)

        y_pred = np.argmax(pred, axis=-1) * flags_down
        

        #    voxel_export("./blender/suncg_R_UNET_E"+str(i)+".bin", y_pred, shape=(60, 36, 60))
        #    print("nyu_R_UNET_E_Pred")
        out_file = out_prefix+'_view'+str(view)+'_prediction.bin'
        voxel_export(out_file, y_pred, shape=prediction_shape)

        out_file = out_prefix+'_view'+str(view)+'_flags.bin'
        voxel_export(out_file, flags_down, shape=prediction_shape)

        out_file = os.path.join(out_prefix+'_view'+str(view)+'_surface.bin')
        voxel_export(out_file, vox_grid_down, shape=prediction_shape)

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

    print("                                     \n")
    
    #################################################################################### Start
    #out_file = out_prefix + '_full_surface.bin'
    #voxel_export(out_file, surf_full, surf_full.shape)
    #################################################################################### End


    y_pred = np.argmax(pred_full, axis=-1)
    # fill camera position
    y_pred[zs-4:zs+4,0,xs-4:xs+4] = 2

    # class mappings
    y_pred[y_pred == 6] = 8  # bed -> table
    y_pred[y_pred == 9] = 11  # tv -> objects
    pred_full = to_categorical(y_pred, num_classes=12)
    
    #################################################################################### Start
    #out_file = out_prefix+'_full_prediction.bin'
    #################################################################################### End
    
    
    
    
    ######################################################################################### start
    print("Combining all views...")

    
    if FILTER:
        print("Filtering...")
        y_pred = voxel_filter(y_pred)

    if MIN_VOXELS>1:
        print("Removing small instances (<%d voxels)..." % MIN_VOXELS)
        y_pred = instance_remover(y_pred, min_size=MIN_VOXELS)

    if SMOOTHING:
        print("Smoothing...")
        y_pred = voxel_fill(y_pred)

    if FILL_LIMITS:
        print("Completing room limits...")
        y_pred = fill_limits_vox(y_pred)

    if REMOVE_INTERNAL:
        print("Removing internal voxels of the objects...")
        y_pred = remove_internal_voxels_v2(y_pred, camx, camy, camz, V_UNIT)

    print("           ")

    out_file = out_prefix + '_surface'
    print("Exporting surface to       %s.obj" % out_file)
    obj_export(out_file, surf_full, surf_full.shape, camx, camy, camz, V_UNIT, include_top=INCLUDE_TOP,
                                                                               triangular=TRIANGULAR_FACES)

    out_file = out_prefix+'_prediction'
    print("Exporting prediction to    %s.obj" % out_file)
    obj_export(out_file, y_pred, (xs*2,ys,zs*2), camx, camy, camz, V_UNIT, include_top=INCLUDE_TOP,
                                                                           triangular=TRIANGULAR_FACES,
                                                                           inner_faces=INNER_FACES)

    print("Finished!\n")
    
    ########################################################################################## end
    voxel_export(out_file, y_pred, shape=(zs*2,ys,zs*2))
    shape=(zs*2,ys,zs*2)
    print("Shape:", shape)

    gt_grid = get_gt(gt_point_cloud,
                     min_x=left_dist, max_x=right_dist,
                     min_y=floor_height, max_y=ceil_height,
                     min_z=back_dist, max_z=front_dist)


    out_file = out_prefix + '_GT.bin'
    voxel_export(out_file, gt_grid, gt_grid.shape)

    gt_grid = gt_grid * flags_full

    #comparing
    vox_compare = y_pred.copy()
    vox_compare[(gt_grid>0) & (y_pred==0)] = 14

    out_file = out_prefix+'_compare.bin'
    voxel_export(out_file, vox_compare, shape=(zs*2,ys,zs*2))

    #evaluation

    y_true =  to_categorical(gt_grid)

    comp_inter, comp_union, tp, fp, fn = comp_iou_stanford(y_true, pred_full, surf_full)
    print("Comp IOU: ",  100 * np.mean((comp_inter + 0.00001) / (comp_union + 0.00001)))

    mean_seg_iou = 0
    qt = 0
    for cl in range(0, 11):
        seg_inter, seg_union = seg_iou_stanford(y_true, pred_full, cl + 1)
        if np.sum(y_true[:,:,:,cl+1])>0:
            seg_iou = 100*np.mean((seg_inter+0.00001)/(seg_union+0.00001))
            mean_seg_iou += seg_iou
            qt += 1
            print("Class (%d) %s - Seg IOU: %2.2f  SUM: %f" % (cl+1,  get_class_names()[cl], seg_iou, np.sum(y_true[:, :, :, cl+1])))
          

    mean_seg_iou /= qt
    print("Mean Seg IOU: ", mean_seg_iou)


def parse_arguments():
################################################################################################# Start
    global DATA_PATH, OUTPUT_PATH, BASELINE, V_UNIT, NETWORK, FILTER, SMOOTHING, \
           FILL_LIMITS, MIN_VOXELS, TRIANGULAR_FACES, WEIGHTS_PATH, INCLUDE_TOP, REMOVE_INTERNAL, INNER_FACES
################################################################################################# End
    print("\nSemantic Scene Completion Inference from 360 depth maps\n")

    parser = argparse.ArgumentParser()

    parser.add_argument("area",    help="Area", type=str)
    parser.add_argument("room",    help="360 rgb", type=str)
    parser.add_argument("camera",  help="output file prefix", type=str)
    parser.add_argument("--v_unit",     help="Voxel size. Default 0.02", type=float, default=0.02, required=False)
    parser.add_argument("--network",   help="Network to be used", type=str,
                                       default="EdgeNet", choices=["EdgeNet", "USSCNet"], required=False)
    parser.add_argument("--base_path",   help="Base path", type=str, default=base_path, required=False)
    parser.add_argument("--output_path",   help="Output path", type=str, default=output_path, required=False)
###################################################################################################################  Start
    parser.add_argument("--filter",        help="Apply 3D low-pass filter? Default yes.", type=str,
                                           default="Y", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--smoothing",     help="Apply smoothing (fill small holes)? Default yes.", type=str,
                                           default="Y", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--fill_limits",   help="Fill walls on room limits? Default yes.", type=str,
                                           default="Y", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--remove_internal",   help="Remove internal voxels? Default no.", type=str,
                                           default="N", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--inner_faces",   help="Include inner faces of objects? Default no.", type=str,
                                           default="N", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--min_voxels",    help="Minimum number of voxels per object instance. Default %d."%MIN_VOXELS, type=int,
                                           default=MIN_VOXELS, required=False)
    parser.add_argument("--triangular",    help="Use triangular faces? Default No.", type=str,
                                           default="N", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--include_top",   help="Include top (ceiling) in output model? Default No.", type=str,
                                           default="Y", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--weights_path",   help="Weights path. Default %s"%WEIGHTS_PATH, type=str,
                                           default=WEIGHTS_PATH, required=False)
##################################################################################################################### End
    args = parser.parse_args()
##################################################################################################################### Start
    FILTER = args.filter in ["Y", "y"]
    SMOOTHING = args.smoothing in ["Y", "y"]
    REMOVE_INTERNAL = args.remove_internal in ["Y", "y"]
    FILL_LIMITS = args.fill_limits in ["Y", "y"]
    INNER_FACES = args.inner_faces in ["Y", "y"]
    MIN_VOXELS = args.min_voxels
    TRIANGULAR_FACES = args.triangular in ["Y", "y"]
    INCLUDE_TOP = args.include_top in ["Y", "y"]
    WEIGHTS_PATH = args.weights_path
################################################################################################################### End

    depth_map = os.path.join(args.base_path, args.area, 'camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_sdepth.png')
    rgb_file = os.path.join(args.base_path, args.area, 'camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_srgb.png')
    gt_file = os.path.join(args.base_path, args.area, 'camera_'+args.camera+'_'+args.room+'_frame_equirectangular_domain_gt.pkl')
    
    network = args.network
    output = os.path.join(args.output_path, args.area + '_'+args.camera+'_'+args.room+'_'+network)

    args = parser.parse_args()

    v_unit = args.v_unit

    print("360 depth map:", depth_map)
    print("Output:", output)
    print("V_Unit:", v_unit)
    print("")



    return depth_map, rgb_file, gt_file, output, v_unit, network

# Main Function
def Run():
    depth_map, rgb_file, gt_file, output, v_unit, network = parse_arguments()
    process(depth_map, rgb_file, gt_file, output, v_unit, network)


if __name__ == '__main__':
  Run()
