import cv2
import cv
import matplotlib.pyplot as plt
import vgg_face_net
import tensorflow as tf
import numpy as np
import sklearn
import time
from vgg_face_net import VggFace
import sklearn.metrics.pairwise as pw

mean_data=np.array([129.1863, 104.7624, 93.5940])
'''
face_box = (429.04, 131.80, 480.15, 182.92)
image = cv2.imread('data/A.J._Buckley_977.jpg')
imageCopy = image[132:183, 430:480,:]
cv2.imshow('',imageCopy)

face_box = (65.00, 65.00, 210.18, 210.18)
image = cv2.imread('data/120210-GF233.JPG')
imageCopy = image[65:210, 65:210,:]

face_box = ( 75.76, 60.67, 211.19, 196.10)
image = cv2.imread('data/CODonnell121709-S159.jpg')
image_head = image[61:196,76:211,:]



#face_box = (165.21 105.50 298.57 238.86 )
image = cv2.imread('data/a.j_buckley_2706152.jpg')
image_head = image[105:238,165:298,:]
'''
#171.87 37.09 281.81 147.03
image = cv2.imread('data/20876524.jpg')
image_head = image[37:147,171:281,:]
imageCopy = cv2.resize(image_head,(224, 224))
imageCopy = imageCopy.astype(np.float32)
imageCopy -= mean_data
imageCopy = imageCopy.reshape([1,224,224,3])

x = tf.placeholder(dtype=tf.float32, shape=[1,224,224,3])
net = VggFace(x, 2622,[])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    net.load_initial_weights(sess)
    start = time.time()
    prob = sess.run(net.fc8, feed_dict={x:imageCopy})
    print("comsume {}".format(time.time() - start))
    idx = np.argmax(prob)
    #print(prob)
    print("idx:{}".format(idx))


cv2.imshow('', image_head)
cv2.waitKey(0)
#dis = pw.pairwise_distances(feature, feature1, metric='cosine')
pw.pairwise_distances(np.array([1,2,3]), np.array([1,2,3]),metric='cosine')

