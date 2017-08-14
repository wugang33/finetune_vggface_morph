import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from alexnet import AlexNet
from caffe_classes import class_names
import tensorflow as tf

mean = np.array([104., 117., 124.], dtype=np.float32)
current_dir = os.getcwd()
images = [os.path.join(current_dir,'images', file) for file in os.listdir('images') if file.endswith('jpeg')]

imgs = [cv2.imread(image_file) for image_file in images]
'''
figure = plt.figure(figsize=(15,6))

for i,img in enumerate(imgs):
    figure.add_subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
'''
x = tf.placeholder(dtype=tf.float32, shape=[1, 227, 227, 3])
keep_prob = tf.placeholder(dtype=tf.float32)
model = AlexNet(x, keep_prob, 1000, [])
score = model.fc8
softmax = tf.nn.softmax(score)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    var = tf.trainable_variables()
    model.load_initial_weights(sess)
    avar = tf.trainable_variables()
    figure2 = plt.figure(figsize=(15, 6))
    for i, image in enumerate(imgs):
        img = cv2.resize(image.astype(np.float32), (227, 227))
        img -= mean
        img = img.reshape((1, 227, 227, 3))
        probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
        class_name = class_names[np.argmax(probs)]
        figure2.add_subplot(1, 3, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.imshow(image)
        plt.title("Class: " + class_name + ", probability: %.4f" % probs[0, np.argmax(probs)])
        plt.axis('off')

plt.waitforbuttonpress()