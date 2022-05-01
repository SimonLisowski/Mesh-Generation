import os
#Path configurations

#Default settings
SUNCG_TRAIN = '/d02/data/csscnet_edges_preproc/SUNCGtrain'
SUNCG_TEST  = '/d02/data/csscnet_edges_preproc/SUNCGtest_49700_49884'
SUNCG_EVAL  = '/home/adn/sscnet/data/eval/SUNCGtest_49700_49884'

NYU_TRAIN   = '/d02/data/csscnet_edges_preproc/NYUtrain'
NYU_TEST    = '/d02/data/csscnet_edges_preproc/NYUtest'
NYU_EVAL    = '/d02/data/NYU/NYUeval'

GEN_TRAIN = '/d02/data/csscnet_edges_preproc/SUNCGGENtrain'
GEN_TEST  = '/d02/data/csscnet_edges_preproc/SUNCGGENtest'
GEN_EVAL  = 'none'


def read_config(dataset, mat=False):
    global SUNCG_TRAIN, SUNCG_TEST, SUNCG_EVAL, NYU_TRAIN, NYU_TEST, NYU_EVAL, GEN_TRAIN, GEN_TEST, GEN_EVAL

    try:
        path_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "paths.conf")
        f = open(path_file, 'r')
        print('Using path config:',path_file)
    except:
        print('paths.conf file not found, using defaults')
        if dataset == 'SUNCG':
            if mat:
                return SUNCG_TRAIN, SUNCG_TEST, SUNCG_EVAL
            else:
                return SUNCG_TRAIN, SUNCG_TEST
        elif dataset == 'NYU':
            if mat:
                return NYU_TRAIN, NYU_TEST, NYU_EVAL
            else:
                return NYU_TRAIN, NYU_TEST
        elif dataset == 'GEN':
                if mat:
                    print('GEN dataset has no mat')
                    exit(-1)
                else:
                    return GEN_TRAIN, GEN_TEST
        else:
            print('Invalid dataset', dataset)
            exit(-1)

    with(f) as file:
        for line in file:
            ln = line.split()
            if ( len(ln)==0) or (ln[0][0:1]  == '#'):
                continue
            if  (len(ln)==2) or (len(ln) > 2 and ln[0][0:1]  == '#'):

                if ln[0]=='SUNCG_TRAIN':
                    SUNCG_TRAIN = ln[1]
                elif  ln[0]=='SUNCG_TEST':
                    SUNCG_TEST = ln[1]
                elif  ln[0]=='SUNCG_EVAL':
                    SUNCG_EVAL = ln[1]
                elif  ln[0]=='NYU_TRAIN':
                    NYU_TRAIN = ln[1]
                elif  ln[0]=='NYU_TEST':
                    NYU_TEST = ln[1]
                elif ln[0] == 'NYU_EVAL':
                    NYU_EVAL = ln[1]
                elif ln[0] == 'GEN_TRAIN':
                    GEN_TRAIN = ln[1]
                elif ln[0] == 'GEN_TEST':
                    GEN_TEST = ln[1]
                elif ln[0] == 'GEN_EVAL':
                    GEN_EVAL = ln[1]
                else:
                    print ('Error in config file:', ln)
                    exit(-1)
                print ('%-15s %s' % (ln[0]+':',ln[1]))
            else:
                print ('Error in config file:', ln)
                exit(-1)
    if dataset == 'SUNCG':
        if mat:
            return SUNCG_TRAIN, SUNCG_TEST, SUNCG_EVAL
        else:
            return SUNCG_TRAIN, SUNCG_TEST
    elif dataset == 'NYU':
        if mat:
            return NYU_TRAIN, NYU_TEST, NYU_EVAL
        else:
            return NYU_TRAIN, NYU_TEST
    elif dataset == 'GEN':
            if mat:
                print('GEN dataset has no mat')
                exit(-1)
            else:
                return GEN_TRAIN, GEN_TEST
    else:
        print('Invalid dataset', dataset)
        exit(-1)
