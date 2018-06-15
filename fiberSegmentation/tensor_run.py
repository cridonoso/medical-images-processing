import tensorflow as tf
from read_write_bundles import read_bundle

path = 'atlas_faisceaux_MNI/atlas_AR_ANT_LEFT_MNI.bundles'
path2 = 'whole_brain_MNI_100k_21p.bundles'


fiberO = tf.placeholder(tf.float32, shape=[21, 3])
fiberC = tf.placeholder(tf.float32, shape=[21, 3])
fiberC_inv = tf.placeholder(tf.float32, shape=[21, 3])

results_1 = []
results_2 = []
for k in range(0,21):
    distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(fiberO[k], fiberC[k]))))
    distance_inv = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(fiberO[k], fiberC_inv[k]))))
    results_1.append(distance)
    results_2.append(distance_inv)

max_distance_1 = tf.reduce_max(results_1)
max_distance_2 = tf.reduce_max(results_2)

max1 = tf.placeholder(tf.float32)
max2 = tf.placeholder(tf.float32)
min = tf.arg_min([max1,max2], 0)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start segmentation
with tf.Session() as sess:
    sess.run(init)
    # Reading data
    whole_brain = read_bundle(path2)
    object_ = read_bundle(path)
    th = 5000
    umbral = 9
    for object in object_:
        for brain in whole_brain[0:th]:
            r, r2 = sess.run([max_distance_1, max_distance_2],
                             feed_dict={fiberO: object,
                                        fiberC: brain,
                                        fiberC_inv: brain[::-1]})
            two_dist = (r,r2)
            m = sess.run(min,
                         feed_dict={max1: r,
                                    max2: r2
                                    })
            if two_dist[m]<=umbral:
                print two_dist[m]

