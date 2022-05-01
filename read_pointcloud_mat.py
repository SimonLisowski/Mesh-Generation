import numpy as np
from lib_edgenet360.py_cuda import lib_edgenet360_setup
import os
import glob
import json
import pandas as pd

area_dir='area_3' #desired area
base_dir = './Data/stanford' #2D-3D Semantics dataset root
output_dir = './Data/stanford_processed' #folder to put extracted groung truth

sscnet_classes = { "ceil":1, "floor":2, "wall":3, "wind":4,
                   "chair":5, "bed":6, "sofa":7, "table":8, "tvs":9, "furn":10, "objs":11
}

stanford_classes = {
              'ceiling': sscnet_classes['ceil'],
              'floor': sscnet_classes['floor'],
              'wall': sscnet_classes['wall'],
              'beam': sscnet_classes['wall'],
              'column': sscnet_classes['wall'],
              'window': sscnet_classes['wind'],
              'door': sscnet_classes['wall'],
              'table': sscnet_classes['table'],
              'chair': sscnet_classes['chair'],
              'sofa': sscnet_classes['sofa'],
              'bookcase': sscnet_classes['furn'],
              'board': sscnet_classes['objs'],
              'clutter': sscnet_classes['objs']
}

PI = 3.14159265
DISP_SCALE = 2.0
DISP_OFFSET = -120.0

baseline=0.264

lib_edgenet360_setup(device=0, num_threads=1024, v_unit=0.021, v_margin=0.24, f=518.8579, debug=0)


mat_file = os.path.join(base_dir,area_dir,'./3d/pointcloud.mat')
pose_path = os.path.join(base_dir,area_dir,'./pano/pose')

def get_str_from_array(char_array):
    return''.join([chr(v[0]) for v in char_array])

def get_files_from_path(data_path, criteria):
    return glob.glob(os.path.join(data_path, criteria))


import h5py

f = h5py.File(mat_file)
area = list(f.keys())[1]
print("Area Name:", area)


area_ds =  f[area]
alignment_ds = area_ds['Disjoint_Space/AlignmentAngle']
#alignment = [f[alignment_ds[ix,0]].value[0,0] for ix in range(alignment_ds.shape[0])  ]
alignment = [f[alignment_ds[ix,0]][()] for ix in range(alignment_ds.shape[0])  ]

name_ds = area_ds['Disjoint_Space/name']
#names_array = [ f[name_ds[ix,0]].value    for ix in range(name_ds.shape[0])  ]
names_array = [ f[name_ds[ix,0]][()]    for ix in range(name_ds.shape[0])  ]

names = [get_str_from_array(x) for x in names_array]

objects_ds = area_ds['Disjoint_Space/object']


for obj_ix  in list(range(objects_ds.shape[0])):

  print("    Room %d/%d %s..." % (obj_ix+1, objects_ds.shape[0], names[obj_ix]), end= " ")
  
  pose_search = 'camera_*_' + names[obj_ix] + '_frame_equirectangular_domain_pose.json'
  pose_files = get_files_from_path(pose_path, pose_search)

  if len(pose_files)==0:
      print("No cameras!")
      continue


  print("%d cameras..." % len(pose_files))

  object = f[objects_ds[obj_ix,0]]

  room_objs_ds = object['global_name']
  print("        %d objects..." % room_objs_ds.shape[0])

  #room_obj_names_array = [f[room_objs_ds[ix, 0]].value for ix in range(room_objs_ds.shape[0])]
  room_obj_names_array = [f[room_objs_ds[ix, 0]][()] for ix in range(room_objs_ds.shape[0])]
  try:
      room_obj_names = [ get_str_from_array(x) for x in room_obj_names_array ]
  except:
      print("Error objects names...")
      continue

  print("        objects names OK...")

  room_obj_points_ds =   object['points']
  #room_obj_points_array = [f[room_obj_points_ds[ix, 0]].value for ix in range(room_obj_points_ds.shape[0])]
  room_obj_points_array = [f[room_obj_points_ds[ix, 0]][()] for ix in range(room_obj_points_ds.shape[0])]

  print("        objects points OK...")


  for pose_ix,pose_file in enumerate(pose_files):
      print("            Camera %d/%d %s ..." %(pose_ix+1,len(pose_files), os.path.basename(pose_file)))

      # Read JSON data into the datastore variable
      with open(pose_file, 'r') as jf:
              datastore = json.load(jf)

      x_loc, y_loc, z_loc = datastore["camera_location"]

      x_out, y_out, z_out = [], [], []
      labels_out = []

      for i,pt in enumerate(room_obj_points_array):
           class_label = room_obj_names[i].split("_")[0]
           #print("Room", obj_ix, names[obj_ix],  "Obj", i, room_obj_names[i], class_label, stanford_classes[class_label])
           pt_ar = np.array(pt)
           x_out.extend(pt_ar[0]-x_loc)
           y_out.extend(pt_ar[1]-y_loc)
           z_out.extend(pt_ar[2]-z_loc)
           lbl = np.zeros((pt_ar.shape[1],))
           lbl[:] = stanford_classes[class_label]
           labels_out.extend(lbl)
      df = pd.DataFrame({'x':np.array(y_out),'y':z_out,'z':-np.array(x_out),'lbl':np.array(labels_out)})
      os.makedirs(os.path.join(output_dir, area_dir),exist_ok=True)
      out_file = os.path.join(output_dir, area_dir, os.path.basename(pose_file)[:-9]+'gt.pkl')
      print("                OK - %d points written" % len(x_out))
      df.to_pickle(out_file)

