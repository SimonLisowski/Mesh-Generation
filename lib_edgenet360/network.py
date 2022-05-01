from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, Add, MaxPooling3D, Activation, BatchNormalization, UpSampling3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.initializers import RandomNormal

def get_sscnet_trunk(x):
    # Conv1

    conv1 = Conv3D(16, 7, strides=2, dilation_rate=1, padding='same', name='conv_1_1', activation='relu')(x)
    conv1 = Conv3D(32, 3, strides=1, dilation_rate=1, padding='same', name='conv_1_2', activation='relu')(conv1)
    conv1 = Conv3D(32, 3, strides=1, dilation_rate=1, padding='same', name='conv_1_3')(conv1)


    add1 = Conv3D(32, 1, strides=1, dilation_rate=1, padding='same', name='red_1')(conv1) #reduction
    add1 = Add()([conv1, add1])
    add1 = Activation('relu')(add1)

    pool1 = MaxPooling3D(2, strides=2)(add1)

    # Conv2

    conv2 = Conv3D(64, 3, strides=1, dilation_rate=1, padding='same', name='conv_2_1', activation='relu')(pool1)
    conv2 = Conv3D(64, 3, strides=1, dilation_rate=1, padding='same', name='conv_2_2', activation='relu')(conv2)

    add2 = Conv3D(64, 1, strides=1, dilation_rate=1, padding='same', name='red_2')(pool1) #reduction
    add2 = Add()([conv2, add2])
    add2 = Activation('relu')(add2)
    add2 = Activation('relu')(add2) # 2 ativações?

    # Conv3

    conv3 = Conv3D(64, 3, strides=1, dilation_rate=1, padding='same', name='conv_3_1', activation='relu')(add2)
    conv3 = Conv3D(64, 3, strides=1, dilation_rate=1, padding='same', name='conv_3_2', activation='relu')(conv3)

    add3 = Add()([conv3, add2])
    add3 = Activation('relu')(add3)
    add3 = Activation('relu')(add3) # 2 ativações?

    # Dilated1

    dil1 = Conv3D(64, 3, strides=1, dilation_rate=2, padding='same', name='dil_1_1', activation='relu')(add3)
    dil1 = Conv3D(64, 3, strides=1, dilation_rate=2, padding='same', name='dil_1_2', activation='relu')(dil1)

    add4 = Add()([dil1, add3])
    add4 = Activation('relu')(add4)
    add4 = Activation('relu')(add4)

    # Dilated2

    dil2 = Conv3D(64, 3, strides=1, dilation_rate=2, padding='same', name='dil_2_1', activation='relu')(add4)
    dil2 = Conv3D(64, 3, strides=1, dilation_rate=2, padding='same', name='dil_2_2', activation='relu')(dil2)

    add5 = Add()([dil2, add4])
    add5 = Activation('relu')(add5)

    # Concat

    conc = concatenate([add2, add3, add4, add5], axis=4)

    # Final Convolutions

    init1 = RandomNormal(mean=0.0, stddev=0.01, seed=None)
    fin = Conv3D(128, 1, padding='same', name='fin_1', activation='relu', kernel_initializer=init1 )(conc)

    init2 = RandomNormal(mean=0.0, stddev=0.01, seed=None)
    fin = Conv3D(128, 1, padding='same', name='fin_2', activation='relu', kernel_initializer=init2 )(fin)

    init3 = RandomNormal(mean=0.0, stddev=0.01, seed=None)
    fin = Conv3D(12, 1, padding='same', name='fin_3', activation='softmax', kernel_initializer=init3 )(fin)

    return fin


def get_sscnet():
    input_tsdf = Input(shape=(240, 144, 240, 1)) #channels last

    fin = get_sscnet_trunk(input_tsdf)

    model = Model(inputs=input_tsdf, outputs=fin)

    return model

def get_csscnet():
    input_tsdf = Input(shape=(240, 144, 240, 1)) #channels last
    input_rgb = Input(shape=(240, 144, 240, 3)) #channels last

    x = concatenate([input_tsdf,input_rgb],axis=-1)

    fin = get_sscnet_trunk(x)

    model = Model(inputs=[input_tsdf,input_rgb], outputs=fin)

    return model

def get_sscnet_edges():
    input_tsdf = Input(shape=(240, 144, 240, 1))
    input_edges = Input(shape=(240, 144, 240, 1))

    x = concatenate([input_tsdf, input_edges], axis=-1)

    fin = get_sscnet_trunk(x)

    model = Model(inputs=[input_tsdf, input_edges], outputs=fin)

    return model

def get_unet_u(x):
    down3 = Conv3D(32, (3, 3, 3), padding='same')(x)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv3D(32, (3, 3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down3)
    # 30 x 18 x 30

    down4 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=2)(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=2)(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down4)
    # 15 x 9 x 15

    center = Conv3D(128, (3, 3, 3), padding='same', dilation_rate=2)(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv3D(128, (3, 3, 3), padding='same', dilation_rate=2)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling3D((2, 2, 2))(center)
    up4 = concatenate([down4, up4], axis=-1)
    up4 = Conv3D(64, (3, 3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv3D(64, (3, 3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv3D(164, (3, 3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 30 x 18 x 30

    up3 = UpSampling3D((2, 2, 2))(up4)
    up3 = concatenate([down3, up3], axis=-1)
    up3 = Conv3D(32, (3, 3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv3D(32, (3, 3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv3D(32, (3, 3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 60 x 36 60

    fin = concatenate([x, up3], axis=-1)
    fin = Conv3D(16, 1, padding='same', name='fin_1', activation='relu')(fin)
    fin = Conv3D(16, 1, padding='same', name='fin_2', activation='relu')(fin)
    fin = Conv3D(12, 1, padding='same', name='fin_3', activation='softmax' )(fin)

    return fin

def get_res_unet_u(y):

    x = Conv3D(32, (3, 3, 3), padding='same')(y)

    down3 = BatchNormalization()(x)
    down3 = Activation('relu')(down3)
    down3 = Conv3D(32, (3, 3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv3D(32, (3, 3, 3), padding='same')(down3)
    down3 = Add()([x,down3])
    down3_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down3)
    # 30 x 18 x 30

    x = Conv3D(64, (3, 3, 3), padding='same')(down3_pool)

    down4 = BatchNormalization()(x)
    down4 = Activation('relu')(down4)
    down4 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=2)(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=2)(down4)
    res = Add()([x,down4])

    down4 = BatchNormalization()(res)
    down4 = Activation('relu')(down4)
    down4 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=2)(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=2)(down4)
    down4 = Add()([res,down4])

    down4_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down4)
    # 15 x 9 x 15

    x = Conv3D(128, (3, 3, 3), padding='same')(down4_pool)

    center = BatchNormalization()(x)
    center = Activation('relu')(center)
    center = Conv3D(128, (3, 3, 3), padding='same', dilation_rate=2)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv3D(128, (3, 3, 3), padding='same', dilation_rate=2)(center)
    res = Add()([x,center])

    center = Activation('relu')(res)
    center = Conv3D(128, (3, 3, 3), padding='same', dilation_rate=2)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv3D(128, (3, 3, 3), padding='same', dilation_rate=2)(center)
    center = Add()([res,center])
    # center

    up4 = Conv3DTranspose(64, (2, 2, 2), strides=(2,2,2))(center)
    res = concatenate([down4, up4], axis=-1)

    res = Conv3D(64, (3, 3, 3), padding='same')(res)

    up4 = BatchNormalization()(res)
    up4 = Activation('relu')(up4)
    up4 = Conv3D(64, (3, 3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv3D(64, (3, 3, 3), padding='same')(up4)
    res = Add() ([res,up4])

    up4 = BatchNormalization()(res)
    up4 = Activation('relu')(up4)
    up4 = Conv3D(64, (3, 3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv3D(64, (3, 3, 3), padding='same')(up4)
    up4 = Add() ([res,up4])
    # 30 x 18 x 30

    up3 = Conv3DTranspose(32, (2, 2, 2), strides=(2,2,2))(up4)
    res = concatenate([down3, up3], axis=-1)

    res = Conv3D(32, (3, 3, 3), padding='same')(res)

    up3 = BatchNormalization()(res)
    up3 = Activation('relu')(up3)
    up3 = Conv3D(32, (3, 3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv3D(32, (3, 3, 3), padding='same')(up3)
    up3 = Add() ([res,up3])

    # 60 x 36 60

    fin = concatenate([y, up3], axis=-1)
    fin = Conv3D(16, 1, padding='same', name='fin_1', activation='relu')(fin)
    fin = Conv3D(16, 1, padding='same', name='fin_2', activation='relu')(fin)
    fin = Conv3D(12, 1, padding='same', name='fin_3', activation='softmax' )(fin)

    return fin



def get_unet_trunk(x):
    down1 = Conv3D(8, (3, 3, 3), padding='same')(x)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv3D(8, (3, 3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down1)
    # 120 x 72 x 120


    down2 = Conv3D(16, (3, 3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv3D(16, (3, 3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down2)
    # 60 x 36 x 60

    return get_unet_u(down2_pool)

def get_res_unet_trunk(x):

    x = Conv3D(8, (3, 3, 3), padding='same', name="C0_tr_input")(x)

    down1 = BatchNormalization(name="BN11_tr_input")(x)
    down1 = Activation('relu',name="AC11_tr_input")(down1)
    down1 = Conv3D(8, (3, 3, 3), padding='same',name="C11_tr_input")(down1)
    down1 = BatchNormalization(name="BN12_tr_input")(down1)
    down1 = Activation('relu',name="AC12_tr_input")(down1)
    down1 = Conv3D(8, (3, 3, 3), padding='same',name="C12_tr_input")(down1)
    down1 = Add(name="ADD1_tr_input")([x,down1])
    down1_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),name="MP1_tr_input")(down1)
    # 120 x 72 x 120


    x = Conv3D(16, (3, 3, 3), padding='same')(down1_pool)

    down2 = BatchNormalization(name="BN21_tr_input")(x)
    down2 = Activation('relu',name="AC21_tr_input")(down2)
    down2 = Conv3D(16, (3, 3, 3), padding='same',name="C21_tr_input")(down2)
    down2 = BatchNormalization(name="BN22_tr_input")(down2)
    down2 = Activation('relu',name="AC22_tr_input")(down2)
    down2 = Conv3D(16, (3, 3, 3), padding='same',name="C22_tr_input")(down2)
    down2 = Add(name="ADD2_tr_input")([x,down2])
    down2_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2),name="MP2_tr_input")(down2)
    # 60 x 36 x 60

    return get_res_unet_u(down2_pool)


def get_unetv2_trunk(x):
    down1 = Conv3D(8, (3, 3, 3), padding='same')(x)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv3D(8, (3, 3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down1)
    # 120 x 72 x 120

    down2 = Conv3D(16, (3, 3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv3D(16, (3, 3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down2)
    # 60 x 36 x 60

    down3 = Conv3D(32, (3, 3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv3D(32, (3, 3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down3)
    # 30 x 18 x 30

    down4 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=2)(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=2)(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down4)
    # 15 x 9 x 15

    center = Conv3D(128, (3, 3, 3), padding='same', dilation_rate=2)(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv3D(128, (3, 3, 3), padding='same', dilation_rate=2)(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling3D((2, 2, 2))(center)
    up4 = concatenate([down4, up4], axis=-1)
    up4 = Conv3D(64, (3, 3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv3D(64, (3, 3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 30 x 18 x 30

    up3 = UpSampling3D((2, 2, 2))(up4)
    up3 = concatenate([down3, up3], axis=-1)
    up3 = Conv3D(32, (3, 3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv3D(32, (3, 3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 60 x 36 60

    fin = concatenate([down2_pool, up3], axis=-1)
    fin = Conv3D(16, 1, padding='same', name='fin_1', activation='relu')(fin)
    fin = Conv3D(16, 1, padding='same', name='fin_2', activation='relu')(fin)
    fin = Conv3D(12, 1, padding='same', name='fin_3', activation='softmax' )(fin)

    return fin


def get_usscnet():
    input_tsdf = Input(shape=(240, 144, 240, 1))

    fin = get_unet_trunk(input_tsdf)

    model = Model(inputs=input_tsdf, outputs=fin)

    return model

def get_ucsscnet():
    input_tsdf = Input(shape=(240, 144, 240, 1)) #channels last
    input_rgb = Input(shape=(240, 144, 240, 3)) #channels last

    x = concatenate([input_tsdf,input_rgb],axis=-1)

    fin = get_unet_trunk(x)

    model = Model(inputs=[input_tsdf,input_rgb], outputs=fin)

    return model


def get_ucsscnet_edges():
    input_tsdf = Input(shape=(240, 144, 240, 1))
    input_edges = Input(shape=(240, 144, 240, 1))

    x = concatenate([input_tsdf, input_edges], axis=-1)

    fin = get_unet_trunk(x)

    model = Model(inputs=[input_tsdf, input_edges], outputs=fin)

    return model

def get_res_unet_edges():
    input_tsdf = Input(shape=(240, 144, 240, 1))
    input_edges = Input(shape=(240, 144, 240, 1))

    x = concatenate([input_tsdf, input_edges], axis=-1)

    fin = get_res_unet_trunk(x)

    model = Model(inputs=[input_tsdf, input_edges], outputs=fin)

    return model

def get_res_unet():
    input_tsdf = Input(shape=(240, 144, 240, 1))

    fin = get_res_unet_trunk(input_tsdf)

    model = Model(inputs=input_tsdf, outputs=fin)

    return model


def get_input_branch(x):
    down1 = Conv3D(8, (3, 3, 3), padding='same')(x)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down1)
    # 120 x 72 x 120


    down2 = Conv3D(16, (3, 3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down2)
    # 60 x 36 x 60

    return down2_pool

def get_ucsscnet_edges_double():
    input_tsdf = Input(shape=(240, 144, 240, 1))
    input_edges = Input(shape=(240, 144, 240, 1))

    i_depth = get_input_branch(input_tsdf)

    i_edges = get_input_branch(input_edges)

    x = concatenate([i_depth, i_edges], axis=-1)

    fin = get_unet_u(x)

    model = Model(inputs=[input_tsdf, input_edges], outputs=fin)

    return model


def get_ucsscnet_edges_v2():
    input_tsdf = Input(shape=(240, 144, 240, 1))
    input_edges = Input(shape=(240, 144, 240, 1))

    x = concatenate([input_tsdf, input_edges], axis=-1)

    fin = get_unetv2_trunk(x)

    model = Model(inputs=[input_tsdf, input_edges], outputs=fin)

    return model



def get_network_by_name(name):

    if name == 'SSCNET':
        return get_sscnet(), 'depth'
    elif name == 'SSCNET_C':
        return get_csscnet(), 'rgb'
    elif name == 'SSCNET_E':
        return get_sscnet_edges(), 'edges'
    elif name == 'UNET':
        return get_usscnet(), 'depth'
    elif name == 'UNET_C':
        return get_ucsscnet(), 'rgb'
    elif name == 'UNET_E':
        return get_ucsscnet_edges(), 'edges'
    elif name == 'UNET_E2':
        return get_ucsscnet_edges_v2(), 'edges'
    elif name == 'R_UNET_E':
        return get_res_unet_edges(), 'edges'
    elif name == 'R_UNET':
        return get_res_unet(), 'depth'
    elif name == 'EdgeNet': #alias for R_UNET_E
        return get_res_unet_edges(), 'edges'
    elif name == 'USSCNet': #alias for R_UNET
        return get_res_unet(), 'depth'
    else:
        raise Exception('Invalid network name: {}'.format(name))

def get_net_name_from_w(weights):
    networks = ['SSCNET', 'SSCNET_C', 'SSCNET_E', 'UNET', 'UNET_C', 'UNET_E', 'UNET_E2', 'UNET_DB', 'R_UNET_E', 'R_UNET']

    for net in networks:
        if ((net+'_LR') == (weights[0:len(net)+3])) or ((net+'fine_LR') == (weights[0:len(net)+7])):
            return net

    print("Invalid weight file: ", weights)
    exit(-1)
