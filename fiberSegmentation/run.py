
# coding: utf-8

# In[5]:


from read_write_bundles import *
import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from time import time


# In[24]:


path = 'atlas_faisceaux_MNI/atlas_AR_ANT_LEFT_MNI.bundles'
path2 = 'whole_brain_MNI_100k_21p.bundles'
whole_brain = read_bundle(path2)

object_ = read_bundle(path)
object_ = object_[0:2]


# In[25]:


def calculate_distance(fiber1, fiber2, umbral):
    distances = []
    for k in range(0, fiber1.shape[0]):
        session  = tf.Session()
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(fiber1[k],fiber2[k]))))
        d = session.run(distance)
        if d > umbral:
            return d
        else:
            distances.append(d)
    return np.max(distances)


fiber = []
t0 = time()
r1s = []
for o in object_:
    num_cores = multiprocessing.cpu_count()
    r = Parallel(n_jobs=num_cores)(delayed(calculate_distance)(o,w,9) 
                                  for w in whole_brain)
    r1s.append(r)

r1s = np.array(r1s)  
r2s = []
for o in object_:
    num_cores = multiprocessing.cpu_count()
    r2 = Parallel(n_jobs=num_cores)(delayed(calculate_distance)(o,w[::-1],9) 
                                  for index,(w) in enumerate(whole_brain))
    r2s.append([r2, index])

r2s = np.array(r2s)
for i in range(0, r1s.shape[0]):   
    minimum = min(r1s[i][0],r2s[i][0])
    if minimum <= 9:
        fiber.append(r2s[i][1])

t1 = time()

print ('Algorithms takes %f' %(t1-t0))


