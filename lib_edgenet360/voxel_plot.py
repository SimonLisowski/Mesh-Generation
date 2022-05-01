import matplotlib.pyplot as plt
import numpy as np
#from lib_csscnet.py_cuda import *
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def plot_points(vox_grid, voxel_shape):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3d Depth")
    cmap=['white', 'aqua', 'beige', 'coral', 'gold', 'green', 'grey', 'khaki','magenta','olive', 'orange', 'pink', 'tan' ]


    for cl in range(1,12):

        print(cl)


        x = []
        y = []
        z = []

        for xx in range(voxel_shape[0]):
            for yy in range(voxel_shape[1]):
                for zz in range(voxel_shape[2]):

                   if (vox_grid[xx,yy,zz] == cl):

                    z.append(-int(zz))
                    y.append(int(yy))
                    x.append(-int(xx))


        ax.scatter(x, y, z, c=cmap[cl], s=2)

    ax.view_init(azim=90, elev=-80)


    plt.draw()
    plt.show()

def plot_rgb(vox_rgb, voxel_shape):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3d RGB")


    x = []
    y = []
    z = []
    c = []

    for xx in range(voxel_shape[0]):
        for yy in range(voxel_shape[1]):
            for zz in range(voxel_shape[2]):

               if (vox_rgb[xx,yy,zz, 0] != 0. and vox_rgb[xx,yy,zz, 1] != 0. and vox_rgb[xx,yy,zz, 2] != 0.) :

                z.append(-int(zz))
                y.append(int(yy))
                x.append(-int(xx))
                c.append((vox_rgb[xx,yy,zz,2],vox_rgb[xx,yy,zz,1],vox_rgb[xx,yy,zz,0]))


    ax.scatter(x, y, z, c=c, s=2)

    ax.view_init(azim=90, elev=-80)


    plt.draw()
    plt.show()





def plot_weights(vox_grid, voxel_shape):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Weigths")
    cmap=['white', 'aqua', 'beige', 'coral', 'gold', 'green', 'grey', 'khaki','magenta','olive', 'orange', 'pink', 'tan' ]


    x = []
    y = []
    z = []

    for xx in range(voxel_shape[0]):
        for yy in range(voxel_shape[1]):
            for zz in range(voxel_shape[2]):

               if (vox_grid[xx,yy,zz] == 1):

                z.append(-int(zz))
                y.append(int(yy))
                x.append(-int(xx))


    ax.scatter(x, y, z, c=cmap[1], s=2)

    x = []
    y = []
    z = []

    for xx in range(voxel_shape[0]):
        for yy in range(voxel_shape[1]):
            for zz in range(voxel_shape[2]):

               if (vox_grid[xx,yy,zz] != 1 and vox_grid[xx,yy,zz] != 0):

                z.append(-int(zz))
                y.append(int(yy))
                x.append(-int(xx))


    ax.scatter(x, y, z, c=cmap[8], s=2)
    ax.view_init(azim=90, elev=-80)


    plt.draw()
    plt.show()


def plot_ground_truth(vox_gt):

    cmap=['white', 'aqua', 'beige', 'coral', 'gold', 'lightgreen', 'mediumpurple', 'khaki','magenta','olive', 'orange', 'pink', 'tan' ]
    emap=['silver', 'teal', 'wheat', 'maroon', 'goldenrod', 'green', 'darkorchid', 'peru','hotpink','g', 'darkorange', 'hotpink', 'sienna' ]
    voxel_shape = (vox_gt.shape[0],vox_gt.shape[1],vox_gt.shape[2])

    voxels =np.zeros(voxel_shape,np.bool)
    colors = np.empty(voxels.shape, dtype=object)
    edge_colors = np.empty(voxels.shape, dtype=object)

    for label in range(1,11):
        voxels = voxels | (vox_gt[:,:,:]==label)
        colors[vox_gt[:,:,:]==label] = cmap[label]
        edge_colors[vox_gt[:,:,:]==label] = emap[label]

    voxels = explode(voxels)
    colors = explode(colors)
    edge_colors = explode(edge_colors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(voxels.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.1
    y[:, 0::2, :] += 0.1
    z[:, :, 0::2] += 0.1
    x[1::2, :, :] += 0.9
    y[:, 1::2, :] += 0.9
    z[:, :, 1::2] += 0.9

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.voxels(x, y, z, voxels, facecolors=colors, edgecolors=edge_colors, alpha=.3)

    #ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    patches = []
    for color,ecolor, label in zip(cmap[1:],emap[1:],get_class_names()):
       patches.append(mpatches.Patch(edgecolor=ecolor, facecolor=color, label=label))
    ax.legend(handles=patches)

    ax.view_init(azim=80, elev=-50)
    #ax.view_init(azim=290, elev=110)

    plt.title("Ground Truth")

    plt.draw()

    return plt, ax

def plot_pred(vox_pred, vox_flags, comp =False):

    cmap=['white', 'aqua', 'beige', 'coral', 'gold', 'lightgreen', 'mediumpurple', 'khaki','magenta','olive', 'orange', 'pink', 'tan' ]
    emap=['silver', 'teal', 'wheat', 'maroon', 'goldenrod', 'green', 'darkorchid', 'peru','hotpink','g', 'darkorange', 'hotpink', 'sienna' ]
    voxel_shape = (vox_pred.shape[0],vox_pred.shape[1],vox_pred.shape[2])

    voxels =np.zeros(voxel_shape,np.bool)
    colors = np.empty(voxels.shape, dtype=object)
    edge_colors = np.empty(voxels.shape, dtype=object)

    for label in range(1,12):
        if comp:
            voxels = voxels | ((vox_pred[:,:,:]==label) & (vox_flags[:,:,:]==4))
            colors[(vox_pred[:,:,:]==label) & (vox_flags[:,:,:]==4)] = cmap[label]
            edge_colors[(vox_pred[:,:,:]==label) & (vox_flags[:,:,:]==4)] = emap[label]
        else:
            voxels = voxels | ((vox_pred[:,:,:]==label) & (vox_flags[:,:,:]>=3))
            colors[(vox_pred[:,:,:]==label) & (vox_flags[:,:,:]>=3)] = cmap[label]
            edge_colors[(vox_pred[:,:,:]==label) & (vox_flags[:,:,:]==3)] = emap[label]
            edge_colors[(vox_pred[:,:,:]==label) & (vox_flags[:,:,:]==4)] = emap[0]

    voxels = explode(voxels)
    colors = explode(colors)
    edge_colors = explode(edge_colors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(voxels.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.1
    y[:, 0::2, :] += 0.1
    z[:, :, 0::2] += 0.1
    x[1::2, :, :] += 0.9
    y[:, 1::2, :] += 0.9
    z[:, :, 1::2] += 0.9

    # and plot everything
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.voxels(x, y, z, voxels, facecolors=colors, edgecolors=edge_colors, alpha=.3)

    #ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=80, elev=-50)
    patches = []
    for color,ecolor, label in zip(cmap[1:],emap[1:],get_class_names()):
       patches.append(mpatches.Patch(edgecolor=ecolor, facecolor=color, label=label))
    ax.legend(handles=patches)
    #ax.view_init(azim=290, elev=110)

    plt.title("Prediction")
    plt.draw()

    return plt, ax


def plot_w(vox_gt):

    cmap=['white', 'aqua', 'beige', 'coral', 'gold', 'lightgreen', 'grey', 'khaki','magenta','olive', 'orange', 'pink', 'tan' ]
    emap=['silver', 'teal', 'wheat', 'maroon', 'goldenrod', 'green', 'darkgrey', 'peru','hotpink','g', 'darkorange', 'hotpink', 'sienna' ]
    voxel_shape = (vox_gt.shape[0],vox_gt.shape[1],vox_gt.shape[2])

    voxels =np.zeros(voxel_shape,np.bool)
    colors = np.empty(voxels.shape, dtype=object)
    edge_colors = np.empty(voxels.shape, dtype=object)

    voxels = voxels | (vox_gt[:,:,:]>0)
    colors[(vox_gt[:,:,:]>0)&(vox_gt[:,:,:]!=1) ] = cmap[2]
    edge_colors[(vox_gt[:,:,:]>0)&(vox_gt[:,:,:]!=1) ] = emap[2]

    colors[(vox_gt[:,:,:]==1) ] = cmap[3]
    edge_colors[(vox_gt[:,:,:]==1) ] = emap[3]

    voxels = explode(voxels)
    colors = explode(colors)
    edge_colors = explode(edge_colors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(voxels.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.1
    y[:, 0::2, :] += 0.1
    z[:, :, 0::2] += 0.1
    x[1::2, :, :] += 0.9
    y[:, 1::2, :] += 0.9
    z[:, :, 1::2] += 0.9

    # and plot everything
    fig = plt.figure()
    plt.title("Weights")
    ax = fig.gca(projection='3d')

    ax.voxels(x, y, z, voxels, facecolors=colors, edgecolors=edge_colors, alpha=.3)

    #ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=80, elev=-50)

    plt.show()


    plt.draw()
    plt.show()


def plot_tsdf(vox_tsdf):
    x = []
    y = []
    z = []
    tsdf = []
    voxel_shape = (vox_tsdf.shape[0],vox_tsdf.shape[1],vox_tsdf.shape[2])

    vox_tsdf = vox_tsdf.reshape(voxel_shape)


    for i in range(int(voxel_shape[0] * voxel_shape[1] * voxel_shape[2])):

        if (not( vox_tsdf[i] >-.2 and vox_tsdf[i] < 0.2)):
            zz = np.floor(i / (voxel_shape[0] * voxel_shape[1]))
            yy = np.floor((i - (zz * voxel_shape[0] * voxel_shape[1])) / voxel_shape[0])
            xx = i - (zz * voxel_shape[0] * voxel_shape[1]) - (yy * voxel_shape[0])

            z.append(-int(zz))
            y.append(int(yy))
            x.append(-int(xx))

            tsdf.append(vox_tsdf[i])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("TSDF")

    ax.scatter(x, y, z, c=tsdf, s=2, cmap=plt.cm.get_cmap('RdYlBu'), vmin=-1, vmax=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(azim=80, elev=-80)
    plt.draw()
    plt.show()

def plot_flipped_tsdf(vox_tsdf, azim=-80, elev=-50):
    x = []
    y = []
    z = []
    tsdf = []

    voxel_shape = (vox_tsdf.shape[0],vox_tsdf.shape[1],vox_tsdf.shape[2])

    vox_tsdf = vox_tsdf.reshape(voxel_shape)

    for xx in range(voxel_shape[0]):
        for yy in range(voxel_shape[1]):
            for zz in range(voxel_shape[2]):

                if(vox_tsdf[xx,yy,zz] < -.8 or vox_tsdf[xx,yy,zz] > 0.8) and vox_tsdf[xx,yy,zz]!=1:
                    z.append(int(zz))
                    y.append(int(yy))
                    x.append(int(xx))

                    tsdf.append(-vox_tsdf[xx, yy, zz])

                #if (yy==20):
                #    z.append(int(zz))
                #    y.append(int(yy))
                #    x.append(int(xx))

                #    tsdf.append(-vox_tsdf[xx,yy,zz])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Flipped TSDF")

    ax.scatter(x, y, z, c=tsdf, s=2, cmap=plt.cm.get_cmap('RdYlBu'), vmin=-1, vmax=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(azim=azim, elev=elev)
    plt.draw()
    plt.show()





def plot_voxel_gt(gt, shape, elev=-80):



    colors = ['black', 'rosybrown', 'darksalmon','sienna','darkkhaki', 'olivedrab',
              'darkcyan', 'royalblue', 'darkorchid', 'palevioletred', 'aquamarine', 'gold']
    ecolors = ['gray', 'indianred', 'coral','saddlebrown','olive', 'darkgreen',
              'darkslategray', 'navy', 'purple', 'crimson', 'teal', 'darkgoldenrod']


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(1,12):

        print (colors[i], ecolors[i])
        voxels = np.flip(np.flip(np.swapaxes((gt==i).reshape(shape),0,2),2),0)

        ax.voxels(voxels, facecolors=colors[i], edgecolor=ecolors[i])

    plt.title("Ground Truth")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(azim=90, elev=elev)
    plt.draw()
    plt.show()

def plot_voxel_grid(grid, shape, elev=-80):



    colors = 'silver'
    ecolors = 'gray'

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    voxels = np.flip(np.flip(np.swapaxes((grid==1).reshape(shape),0,2),2),0)

    ax.voxels(voxels, facecolors=colors, edgecolor=ecolors)

    plt.title("Voxel Grid")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(azim=90, elev=elev)
    plt.draw()
    plt.show()


def plot_voxel_ftsdf(ftsdf, shape, elev=-80):

    cmap = plt.cm.get_cmap('RdYlBu')


    def myCmap(a):
        return cmap(a)

    vec_cmap = np.vectorize(myCmap)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    v = ((ftsdf > -.8) & (ftsdf < -0.002)) |  ((ftsdf > 0.002) & (ftsdf < 0.8))
    #v = ((ftsdf < -.80) |  (ftsdf > 0.80))

    voxels = np.flip(np.flip(np.swapaxes(v.reshape(shape),0,2),2),0)

    t = ((ftsdf*(-1) + 1) / 2)
    t = np.flip(np.flip(np.swapaxes(t.reshape(shape),0,2),2),0)

    e = t * .8

    print(max(t[voxels]))
    print(min(t[voxels]))
    print(max(e[voxels]))
    print(min(e[voxels]))

    colors = np.rollaxis(np.array(vec_cmap(t)).reshape((4, shape[0],shape[1], shape[2])),0,4)

    #ecolors = np.rollaxis(np.array(vec_cmap(e)).reshape((4, shape[0],shape[1], shape[2])),0,4)

    print(colors[20,20,20])

    #print(ecolors[20,20,20])

    print (voxels.shape)
    print (colors.shape)

    ax.voxels(voxels, facecolors=colors, edgecolor=None)

    plt.title("Flipped TSDF")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(azim=90, elev=elev)
    plt.draw()
    plt.show()


