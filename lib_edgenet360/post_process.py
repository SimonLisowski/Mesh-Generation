import numpy as np


def voxel_filter(vox_orig):

    sx, sy, sz = vox_orig.shape

    min_supp=2
    filter3d = np.zeros((5, 5, 5), dtype=np.uint8)

    filter3d[0:5,2,2] = 1
    filter3d[2,2,0:5] = 1
    filter3d[2,0:5,2] = 1
    filter3d[2, 2, 2] = 0

    vox=np.zeros((sx+4, sy+4, sz+4),dtype=np.uint8)
    vox[2:-2,2:-2,2:-2] = vox_orig.copy()

    filtered = vox_orig.copy()

    for x in range(sx):
        print("%d%%" % (100 * x/sx), end="\r")
        for y in range(sy):
            for z in range(sz):
                _x, _y, _z = x+2, y+2, z+2
                cl = vox[_x, _y, _z]

                if cl==0:
                    continue

                sub_vol =   vox[_x-2:_x+3,_y-2:_y+3,_z-2:_z+3].copy()
                sub_vol[sub_vol!=cl] = 0
                sub_vol[sub_vol==cl] = 1

                filtered[x,y,z] = vox[_x,_y,_z] * np.clip(np.sum(sub_vol * filter3d)- min_supp -1, 0, 1)
    return filtered


def remove_internal_voxels(vox_orig):

    sx, sy, sz = vox_orig.shape

    vox=np.ones((sx+2, sy+2, sz+2),dtype=np.uint8)
    vox[1:-1,1:-1,1:-1] = np.array(vox_orig>0, dtype=np.uint8)

    filtered = vox_orig.copy()

    for x in range(sx):
        print("%d%%" % (100 * x/sx), end="\r")
        for y in range(sy):
            for z in range(sz):
                _x, _y, _z = x+1, y+1, z+1
                cl = vox[_x, _y, _z]

                if cl==0:
                    continue

                qtty =   np.sum(vox[_x-1:_x+2,_y-1:_y+2,_z-1:_z+2].copy())

                if qtty==27:
                    filtered[x,y,z] = 0
                else:
                    filtered[x,y,z] = vox_orig[x,y,z]
    return filtered

def remove_internal_voxels_v2(vox_orig, camx, camy, camz, v_unit):

    v_unit = v_unit * 4

    sx, sy, sz = vox_orig.shape
    x, y, z = int(sx/2), int(camy//v_unit), int(sz/2)

    instances = np.zeros(vox_orig.shape, dtype=int)

    void = 2**30 -1

    instances [x, y, z] = void
    vox_orig[x, y, z] = 0

    instance_count = 1

    for x in range(sx):
        print("finding void space %d%%"% (100 * x/sx), end="\r" )
        for y in range(sy):
            for z in range(sz):
                cl = vox_orig[x, y, z]
                if cl>0:
                    continue
                x_start, x_end = max(0,x-1), min(sx,x+2)
                y_start, y_end = max(0,y-1), min(sy,y+2)
                z_start, z_end = max(0,z-1), min(sz,z+2)

                sub_vox = vox_orig[x_start:x_end, y_start:y_end, z_start:z_end]
                sub_ins = instances[x_start:x_end, y_start:y_end, z_start:z_end]

                inst = np.unique(sub_ins[sub_vox==cl] )

                if len(inst) == 0 or max(inst)==0:
                    instance_count += 1
                    sub_ins[sub_vox==cl]=instance_count
                else:
                    current_instance = max(inst)
                    sub_ins[sub_vox==cl] = current_instance
                    for inst_merge in inst:
                        if inst_merge==0:
                            continue
                        instances[instances==inst_merge] = current_instance

    print("                           ", end="\r")

    for x in range(sx):
        print("%d%%"% (100 * x/sx), end="\r")
        for y in range(sy):
            for z in range(sz):
                cl = vox_orig[x, y, z]
                if cl==0 or cl==255:
                    continue
                xi, xf = max(0,x-1), min(sx,x+2)
                yi, yf = max(0,y-1), min(sy,y+2)
                zi, zf = max(0,z-1), min(sz,z+2)
                sub_ins = instances[xi:xf,yi:yf,zi:zf].copy()
                qtty =   np.sum(np.array(sub_ins==void,dtype=np.uint8))

                if qtty == 0:
                    vox_orig[x, y, z] = 0

    instances[instances!=void]=0
    instances[instances==void]=1

    return vox_orig, np.array(instances,dtype=np.uint8)



def instance_remover(vox_orig, min_size=15):

    sx, sy, sz = vox_orig.shape

    instances = np.zeros(vox_orig.shape, dtype=int)

    instance_count = 0

    for x in range(sx):
        print("%d%%"% (100 * x/sx), end="\r" )
        for y in range(sy):
            for z in range(sz):
                cl = vox_orig[x, y, z]
                if cl==0:
                    continue
                x_start, x_end = max(0,x-1), min(sx,x+2)
                y_start, y_end = max(0,y-1), min(sy,y+2)
                z_start, z_end = max(0,z-1), min(sz,z+2)

                sub_vox = vox_orig[x_start:x_end, y_start:y_end, z_start:z_end]
                sub_ins = instances[x_start:x_end, y_start:y_end, z_start:z_end]

                inst = np.unique(sub_ins[sub_vox==cl] )

                if len(inst) == 0 or max(inst)==0:
                    instance_count += 1
                    sub_ins[sub_vox==cl]=instance_count
                else:
                    current_instance = max(inst)
                    sub_ins[sub_vox==cl] = current_instance
                    for inst_merge in range(len(inst)-1):
                        if inst_merge==0:
                            continue
                        instances[instances==inst_merge] = current_instance

    for i in range(instance_count+1):
        if i == 0:
            continue
        members = len(instances[instances==i])

        if members<min_size:
            vox_orig[instances==i] = 0

    return vox_orig



def voxel_fill(vox_orig):
    from scipy import stats

    sx, sy, sz = vox_orig.shape

    vox=np.zeros((sx+2, sy+2, sz+2),dtype=np.uint8)
    vox[1:-1,1:-1,1:-1] = vox_orig.copy()

    filtered = vox_orig.copy()

    filter2d = np.zeros((3, 3, 3, 3), dtype=np.uint8)

    filter2d[0,1,:,:] = 1
    filter2d[1,:,1,:] = 1
    filter2d[2,:,:,1] = 1
    filter2d[:,1, 1, 1] = 0

    for x in range(sx):
        print("%d%%"% (100 * x/sx), end="\r" )

        for y in range(sy):
            for z in range(sz):
                _x, _y, _z = x+1, y+1, z+1

                class_v = np.zeros((3,),dtype=np.uint8)
                sup_v = np.zeros((3,),dtype=np.uint8)
                for filter_idx in range(3):
                    sub_vol = vox[_x - 1:_x + 2, _y - 1:_y + 2, _z - 1:_z + 2].copy() * filter2d[filter_idx]
                    sub_vol_flat = sub_vol[sub_vol != 0]
                    if len(sub_vol_flat) == 0:
                        continue
                    cl = stats.mode(sub_vol_flat)[0][0]
                    sub_vol[sub_vol != cl] = 0
                    sub_vol[sub_vol == cl] = 1
                    support = np.sum(sub_vol)
                    if support > 4:
                        class_v[filter_idx] = cl
                        sup_v[filter_idx] = support
                cl =  class_v[np.argmax(sup_v, axis=-1)]
                if cl > 0:
                    filtered[x, y, z] = cl

    return filtered

def fill_limits_vox(vox):
    xs, ys, zs = 0, 0, 0
    xe, ye, ze = np.array(vox.shape) - 1

    for x in range(vox.shape[0]):
        if np.sum(vox[x,:,:])>0:
            xs=x
            break
    for x in range(vox.shape[0]):
        if np.sum(vox[vox.shape[0]-x-1,:,:])>0:
            xe=vox.shape[0]-x-1
            break
    for z in range(vox.shape[2]):
        if np.sum(vox[:,:,z])>0:
            zs=z
            break
    for z in range(vox.shape[2]):
        if np.sum(vox[:,:,vox.shape[2]-z-1])>0:
            ze=vox.shape[2]-z-1
            break
    for y in range(vox.shape[1]):
        if np.sum(vox[:,vox.shape[1]-y-1,:])>0:
            ye=vox.shape[1]-y-1
            break

    vox[xs, 1:ye+1, zs:ze+1][vox[xs, 1:ye+1, zs:ze+1] == 0] = 3
    vox[xe, 1:ye+1, zs:ze+1][vox[xe, 1:ye+1, zs:ze+1] == 0] = 3
    vox[xs:xe+1, 1:ye, zs][vox[xs:xe+1, 1:ye, zs] == 0] = 3
    vox[xs:xe+1, 1:ye, ze][vox[xs:xe+1, 1:ye, ze] == 0] = 3
    vox[xs:xe+1, 0, zs:ze+1][vox[xs:xe+1, 0, zs:ze+1] == 0] = 2
    vox[xs:xe+1, ye, zs:ze+1][vox[xs:xe+1, ye, zs:ze+1] == 0] = 1

    vox[:xs, :, :] = 255
    vox[xe+1:, :, :] = 255
    vox[:, :ys, :] = 255
    vox[:, ye+1:, :] = 255
    vox[:, :, :zs] = 255
    vox[:, :, ze+1:] = 255


    return vox
