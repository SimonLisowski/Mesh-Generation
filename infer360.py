import argparse
import os

import numpy as np

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CV_PI = 3.141592
#f 518.8579

prediction_shape = (60,36,60)
probs_shape = (60,36,60,12)

DATA_PATH = './Data'
OUTPUT_PATH = './Output'
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


def process(depth_file, rgb_file, out_prefix):
    import numpy as np
    from lib_edgenet360.py_cuda import lib_edgenet360_setup, get_point_cloud, \
         get_voxels, downsample_grid, get_ftsdf, downsample_limits
    from lib_edgenet360.file_utils import obj_export
    from lib_edgenet360.network import get_network_by_name
    from lib_edgenet360.metrics import comp_iou, seg_iou
    from lib_edgenet360.losses import weighted_categorical_crossentropy
    from lib_edgenet360.post_process import voxel_filter, voxel_fill, fill_limits_vox, instance_remover,\
                                            remove_internal_voxels_v2

    from tensorflow.keras.optimizers import SGD
    import cv2

    lib_edgenet360_setup(device=0, num_threads=1024, v_unit=V_UNIT, v_margin=0.24, f=518.8579, debug=0)

    print("Processing point cloud...")

    point_cloud, depth_image = get_point_cloud(depth_file, baseline=BASELINE)

    wx, wy, wz, lat, long, wrd = tuple(range(6))


    ceil_height, floor_height = np.max(point_cloud[:,:,wy]), np.min(point_cloud[:,:,wy])
    front_dist, back_dist = np.max(point_cloud[:,:,wz]), np.min(point_cloud[:,:,wz])
    right_dist, left_dist = np.max(point_cloud[:,:,wx]), np.min(point_cloud[:,:,wx])

    print("room height: %2.2f (%2.2f <> %2.2f)" % (ceil_height - floor_height, ceil_height, floor_height))
    print("room width:  %2.2f (%2.2f <> %2.2f)" % ( right_dist - left_dist, right_dist , left_dist))
    print("room length: %2.2f (%2.2f <> %2.2f)" % ( front_dist - back_dist, front_dist , back_dist))

    camx, camy, camz = -left_dist, -floor_height, -back_dist

    bgr_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    edges_image = cv2.Canny(bgr_image,100,200)

    print("\nLoading %s..." % NETWORK)
    model, network_type = get_network_by_name(NETWORK)
    model.compile(optimizer=SGD(lr=0.01, decay=0.005,  momentum=0.9),
                  loss=weighted_categorical_crossentropy
                  ,metrics=[comp_iou, seg_iou]
                  )

    weight_file = {'USSCNet': 'R_UNET_LR0.01_DC0.0005_621-0.69-0.54.hdf5',
                   'EdgeNet': 'R_UNET_E_LR0.01_DC0.0005_4535-0.77-0.55.hdf5'
    }

    model.load_weights(os.path.join(WEIGHTS_PATH,weight_file[NETWORK]))

    xs, ys, zs = prediction_shape

    pred_full = np.zeros((xs*2,ys,xs*2,12), dtype=np.float32)
    flags_full = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
    surf_full = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)

    print("Inferring...")

    for ix in range(8):

        view = ix + 1
        print("view", view, end='\r')

        vox_grid, vox_grid_edges = get_voxels(point_cloud, depth_image.shape, edges_image,
                              min_x=left_dist - 0.05, max_x=right_dist + 0.05,
                              min_y=floor_height - 0.05, max_y=ceil_height + 0.05,
                              min_z=back_dist - 0.05, max_z=front_dist + 0.05,
                              vol_number=view)

        vox_grid_down = downsample_grid(vox_grid)

        vox_tsdf, vox_tsdf_edges, vox_limits = get_ftsdf(depth_image, vox_grid, vox_grid_edges,
                                         min_x=left_dist - 0.05, max_x=right_dist + 0.05,
                                         min_y=floor_height - 0.05, max_y=ceil_height + 0.05,
                                         min_z=back_dist - 0.05, max_z=front_dist + 0.05, baseline=BASELINE, vol_number=view)


        if network_type=='depth':
            x = vox_tsdf.reshape(1,240,144,240,1)
        elif network_type=='edges':
            x = [vox_tsdf.reshape(1,240,144,240,1),vox_tsdf_edges.reshape(1,240,144,240,1)]
        else:
            raise Exception('Invalid network tyoe: {}'.format(network_type))


        pred = model.predict(x=x)

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

        elif view == 4:
            pred_full[:zs, :, xs:] += np.flip(np.swapaxes(fpred,0,2),axis=0)
            surf_full[:zs, :, xs:] |= np.flip(np.swapaxes(vox_grid_down,0,2),axis=0)

        elif view==5:
            pred_full[:zs, :, xs//2:-xs//2] += np.flip(fpred,axis=[0,2])
            surf_full[:zs, :, xs//2:-xs//2] |= np.flip(vox_grid_down,axis=[0,2])
        elif view==6:
            pred_full[:zs, :, :xs] += np.flip(fpred,axis=[0,2])
            surf_full[:zs, :, :xs] |= np.flip(vox_grid_down,axis=[0,2])

        elif view==7:
            pred_full[zs//2:-zs//2, :,:xs ] += np.flip(fpred,axis=[0,2])
            surf_full[zs//2:-zs//2, :,:xs ] |= np.flip(vox_grid_down,axis=[0,2])

        elif view == 8:
            pred_full[zs:, :, :xs] += np.flip(fpred,axis=2)
            surf_full[zs:, :, :xs] |= np.flip(vox_grid_down,axis=2)

    print("Combining all views...")

    y_pred = np.argmax(pred_full, axis=-1)
    # fill camera position
    y_pred[zs-4:zs+4,0,xs-4:xs+4] = 2

    # class mappings
    #y_pred[y_pred == 6] = 8  # bed -> table
    #y_pred[y_pred == 9] = 11  # tv -> objects

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

def parse_arguments():
    global DATA_PATH, OUTPUT_PATH, BASELINE, V_UNIT, NETWORK, FILTER, SMOOTHING, \
           FILL_LIMITS, MIN_VOXELS, TRIANGULAR_FACES, WEIGHTS_PATH, INCLUDE_TOP, REMOVE_INTERNAL, INNER_FACES

    print("\nSemantic Scene Completion Inference from 360 depth maps\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",         help="360 dataset dir", type=str)
    parser.add_argument("depth_map",       help="360 depth map", type=str)
    parser.add_argument("rgb_file",        help="rgb", type=str)
    parser.add_argument("output",          help="output file prefix", type=str)
    parser.add_argument("--baseline",      help="Stereo 360 camera baseline. Default %5.3f"%BASELINE, type=float,
                                           default=BASELINE, required=False)
    parser.add_argument("--v_unit",        help="Voxel size. Default %5.3f" % V_UNIT, type=float,
                                           default=V_UNIT, required=False)
    parser.add_argument("--network",       help="Network to be used. Default %s" % NETWORK, type=str,
                                           default=NETWORK, choices=["EdgeNet", "USSCNet"], required=False)
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
                                           default="N", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--data_path",     help="Data path. Default %s"%DATA_PATH, type=str,
                                           default=DATA_PATH, required=False)
    parser.add_argument("--output_path",   help="Output path. Default %s"%OUTPUT_PATH, type=str,
                                           default=OUTPUT_PATH, required=False)
    parser.add_argument("--weights_path",   help="Weights path. Default %s"%WEIGHTS_PATH, type=str,
                                           default=WEIGHTS_PATH, required=False)

    args = parser.parse_args()

    BASELINE = args.baseline
    V_UNIT = args.v_unit
    NETWORK = args.network
    FILTER = args.filter in ["Y", "y"]
    SMOOTHING = args.smoothing in ["Y", "y"]
    REMOVE_INTERNAL = args.remove_internal in ["Y", "y"]
    FILL_LIMITS = args.fill_limits in ["Y", "y"]
    INNER_FACES = args.inner_faces in ["Y", "y"]
    MIN_VOXELS = args.min_voxels
    TRIANGULAR_FACES = args.triangular in ["Y", "y"]
    INCLUDE_TOP = args.include_top in ["Y", "y"]
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    WEIGHTS_PATH = args.weights_path

    dataset = args.dataset
    depth_map = os.path.join(DATA_PATH, dataset, args.depth_map)
    rgb_file = os.path.join(DATA_PATH, dataset, args.rgb_file)
    output = os.path.join(OUTPUT_PATH, args.output)

    fail = False
    if not os.path.isfile(depth_map):
        print("Depth map file not found:", depth_map)
        fail = True

    if not os.path.isfile(rgb_file):
        print("RGB file not found:", rgb_file )
        fail = True

    if fail:
        print("Exiting...\n")
        exit(0)

    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("360 depth map:", depth_map)
    print("360 rgb:      ", rgb_file)
    print("Output prefix:", output)
    print("Baseline:     ", BASELINE)
    print("V_Unit:       ", V_UNIT)
    print("")

    return depth_map, rgb_file, output

# Main Function
def Run():
    depth_map, rgb_file, output = parse_arguments()
    process(depth_map, rgb_file, output)


if __name__ == '__main__':
  Run()
