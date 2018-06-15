# coding: utf-8
from read_write_bundles import *
import numpy as np
from time import time
from scipy.spatial import distance
#For parallel
import multiprocess as mp


def calculate_distance(*args):
    dies = []
    fiber1 = args[0][0]
    fiber2 = args[0][1]
    umbral = args[0][2]
    for k in range(0, fiber1.shape[0]):
        d = distance.euclidean(fiber1[k], fiber2[k])
        if d <= umbral:
            dies.append(d)
        else:
            return d, False
    return (dies, fiber2), True


if __name__ == '__main__':
    path = 'atlas_faisceaux_MNI/atlas_AR_ANT_LEFT_MNI.bundles'
    path2 = 'whole_brain_MNI_100k_21p.bundles'
    whole_brain = read_bundle(path2)
    object_ = read_bundle(path)


    parallel = False
    max_num = 10000
    whole_brain = whole_brain[0:max_num]
    print 'object: ',len(object_)
    print 'whole_brain: ', len(whole_brain)
    if not parallel:
        print 'Running with Simple Computing'
        t0 = time()
        distances_new = []
        for o in object_:
            for w in whole_brain:
                dist, save = calculate_distance((o,w[::-1],9))
                if save:
                    distances_new.append(dist[1])
                    print dist[1]


        t1 = time()
        print ('Algorithms takes %f' %(t1-t0))
    else:
        print 'Running in Parallel Computing'
        p = mp.Pool()
        t0 = time()
        input = [(o, w[::-1], 9) for o in object_ for w in whole_brain]
        results = p.map(calculate_distance, input)
        p.close()
        p.join()

        for r in results:
            if r[1]:
                print r[0]

        t1 = time()
        print ('Algorithms takes %f' %(t1-t0))
