# coding: utf-8
from read_write_bundles import *
import numpy as np
from time import time
from scipy.spatial import distance
#to save vectors
import pickle
import os



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

    rootdir = './atlas_faisceaux_MNI'
    extensions = ('.bundles')
    t00 = time()
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in extensions:
                print 'reading',file,'...'
                part= file
                path = 'atlas_faisceaux_MNI/'+part
                path2 = 'whole_brain_MNI_100k_21p.bundles'
                whole_brain = read_bundle(path2)
                object_ = read_bundle(path)


                max_num = 10000
                whole_brain = whole_brain[0:max_num]
                print 'object: ',len(object_)
                print 'whole_brain: ', len(whole_brain)

                t0 = time()
                distances_new = []
                indexes = []
                for o in object_:
                    for ind, (w) in enumerate(whole_brain):
                        dist, save = calculate_distance((o,w[::-1],9))
                        dist2, save2 = calculate_distance((o, w, 9))
                        if save and save2:
                            t = (dist[0],dist2[0])
                            to = (dist[1],dist2[1])
                            m = np.argmin(t)
                            distances_new.append(to[m])
                            indexes.append(ind)
                            #print 'both: ',to[m]
                        else:
                            if save:
                                distances_new.append(dist[1])
                                indexes.append(ind)
                                #print 'inverse: ',dist[1]
                            if save2:
                                distances_new.append(dist2[1])
                                indexes.append(ind)
                                #print 'normal',dist2[1]


                t1 = time()
                print part,'takes %f' %(t1-t0)
                name = './saved/'+str(part)+'.pkl'
                with open(name, 'wb') as f:
                    pickle.dump(distances_new, f)
                name2 = './saved/'+str(part)+'_indexex.pkl'
                with open(name2, 'wb') as f:
                    pickle.dump(indexes, f)
                print '='*50
    t11 = time()
    print ('Algorithms takes %f' % (t1 - t0))