import math
import numpy as np
import openmesh as om
from statistics import mean
from skimage import measure
from collections import Counter
from scipy.spatial import KDTree

class_colors = [
    (0.1, 0.1, 0.1, 1),
    (0.0649613, 0.467197, 0.0667303, 1),
    (0.1, 0.847035, 0.1, 1),
    (0.0644802, 0.646941, 0.774265, 1),
    (0.131518, 0.273524, 0.548847, 1),
    (1, 0.813553, 0.0392201, 1),
    (1, 0.490452, 0.0624932, 1),
    (0.657877, 0.0505005, 1, 1),
    (0.0363214, 0.0959549, 0.548847, 1),
    (0.316852, 0.548847, 0.186899, 1),
    (0.548847, 0.143381, 0.0045568, 1),
    (1, 0.241096, 0.718126, 1),
    (0.9, 0.0, 0.0, 1),
    (0.4, 0.0, 0.0, 1),
    (0.3, 0.3, 0.3, 1)
    ]

class_colors_ordered = [
    (0.0644802, 0.646941, 0.774265, 1), #wall
    (0.1, 0.847035, 0.1, 1), #floor
    (0.0649613, 0.467197, 0.0667303, 1), #ceiling
    (0.548847, 0.143381, 0.0045568, 1), #furniture
    (0.316852, 0.548847, 0.186899, 1), #table
    (0.657877, 0.0505005, 1, 1), #sofa
    (1, 0.490452, 0.0624932, 1),  # bed
    (0.0363214, 0.0959549, 0.548847, 1), #tvs
    (1, 0.813553, 0.0392201, 1), #chair
    (1, 0.241096, 0.718126, 1), #objects
    (0.131518, 0.273524, 0.548847, 1), #window
    (0.1, 0.1, 0.1, 1), #empty
    (0.9, 0.0, 0.0, 1), #error1
    (0.4, 0.0, 0.0, 1), #error2
    (0.3, 0.3, 0.3, 1) #error3
    ]

def draw_mesh(name, vert_array, cat_array, voxel_list, v_unit):
    print("Exporting mesh to "+name)

    vu = v_unit * 4

    # Use marching cubes to obtain the surface mesh
    verts, faces, normals, values = measure.marching_cubes(vert_array, spacing=(1.0, 1.0, 1.0))

    mesh = om.TriMesh()

    mesh.request_face_colors()

    vert_arr = np.array([])
    face_arr = np.array([])
    color_arr = np.array([])
    color_counter_arr = np.array([])

    tree = KDTree(voxel_list, balanced_tree=True)

    print("           ")
    print("Adding mesh vertices...")
    for x in range(len(verts)):
        vertex = verts[x]
        vertex = (vertex - 500) * vu
        vh = mesh.add_vertex(vertex)
        vert_arr = np.append(vert_arr, vh)
        query = tree.query(vertex, k=1)
        color_arr = np.append(color_arr, cat_array[query[1]])
        mesh.set_color(vh, class_colors[cat_array[query[1]]])
        print("%d%%" % (100 * x / len(verts)), end="\r")

    print("Adding mesh faces...")
    for y in range(len(faces)):
        face = faces[y]
        fh = mesh.add_face([vert_arr[face[0]], vert_arr[face[1]], vert_arr[face[2]]])
        color_counter = Counter([color_arr[face[0]], color_arr[face[1]], color_arr[face[2]]])
        if len(color_counter) == 1:
            color = color_counter.most_common(1)[0][0]
            mesh.set_color(fh, class_colors[int(color)])
        else:
            for idx in range(len(class_colors_ordered)):
                if class_colors_ordered[idx] in [class_colors[int(index)] for index in set(color_counter)]:
                    mesh.set_color(fh, class_colors_ordered[idx])
                    break
        print("%d%%" % (100 * y / len(faces)), end="\r")
    print("           ")
    
    om.write_mesh(str(name), mesh, face_color=True)

    '''
    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 1000)

    plt.tight_layout()
    plt.show(block=True)
    '''