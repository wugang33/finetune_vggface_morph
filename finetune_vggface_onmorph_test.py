import tensorflow as tf
import os
from vgg_face_net import VggFace
import numpy as np
from dategenerator import ImageDataGenerator

train_file = './morph_train.txt'
val_file = './morph_test.txt'
from datetime import datetime

num_epochs = 30
batch_size = 32

num_classes = 31

filewriter_path = "./finetune_vggface_morph/face"
checkpoint_path = "./finetune_vggface_morph/"

X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
Y = tf.placeholder(dtype=tf.int64, shape=[None, num_classes])
model = VggFace(X, num_classes, [])

score = model.fc8

with tf.variable_scope('accuracy') as scope:
    correct_pend = tf.equal(tf.argmax(score, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pend, tf.float32))

train_data_generator = ImageDataGenerator(train_file, horizontal_flip=True, shuffle=True,
                                          mean=np.array([129.1863, 104.7624, 93.5940]), scale_size=(224, 224),
                                          nb_classes=num_classes)
test_data_generator = ImageDataGenerator(train_file, horizontal_flip=False, shuffle=False,
                                         mean=np.array([129.1863, 104.7624, 93.5940]), scale_size=(224, 224),
                                         nb_classes=num_classes)

train_batches_per_epoch = np.floor(train_data_generator.data_size / batch_size).astype(np.int16)
test_batches_per_epoch = np.floor(test_data_generator.data_size / batch_size).astype(np.int16)

saver = tf.train.Saver()
import cv2
import dlib
detector = dlib.get_frontal_face_detector()
import re
r = r'(\d+)_(\d+)(M|F)(\d+).JPG'
regex = re.compile(r)
def process_image(datapath, imgname, mean=np.array([129.1863, 104.7624, 93.5940])):
    img = cv2.imread(os.path.join(datapath, imgname))
    id, seq, gender, age = regex.findall(imgname)[0]
    if int(age) >46 or int(age) <16:
        return None
    if img is None:
        raise RuntimeError("can not open file:%s"%(os.path.join(datapath, imgname)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img)
    #print("Number of faces detected: {}".format(len(dets)))
    if len(dets) != 1:
        return None
        raise RuntimeError('detect face num:%d'%(len(dets)))
    for i, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        if d.top() < 0 or d.left() < 0 or d.right() < 0 or d.bottom() < 0:
            return None
        if d.bottom() >= img.shape[0] or d.right() >= img.shape[1]:
            return None
        img_croped = img[d.top():d.bottom(), d.left():d.right(), :]
        img_croped = cv2.resize(img_croped, (224, 224))
        #cv2.imshow('w', img_croped)
        #cv2.waitKey(0)
        img_croped = img_croped.astype(np.float32)
        img_croped -= mean
        img_croped = img_croped.reshape((1, 224, 224, 3))
        return img_croped, int(age)
#datapath = '/home/wg/data/morph/MORPH'
import time
datapath = '/home/wg/my_finetune_alexnet/data/my_test'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # model.load_initial_weights(sess)
    saver.restore(sess, 'finetune_vggface_morph/model_epoch15.ckpk')
    print("{}Start to validation...".format(datetime.now()))
    test_acc = 0.
    test_count = 0.
    files = os.listdir(datapath)
    test_age = 0.
    for i,file in enumerate(files):
        back= process_image(datapath, file)
        if back is None:
            continue
        x_xs,age = back
        if age > 46:
            continue;
        if i%100 is 0 and i is not 0:
            print("{} Test validation{:.4f} mae:{:.4f} total_count:{} idx:{}".format(datetime.now(), test_acc/test_count, test_age/test_count, test_count, i))
        start = time.time()
        score_val = sess.run(score, feed_dict={X: x_xs})
        print("consume secons:%f"%(time.time() - start))
        compute_age = np.argmax(score_val, 1)[0] + 16

        #print("compute:%d groud true:%d"%(compute_age, age))
        test_age += (abs(compute_age - age))
        if compute_age == age:
            test_acc += 1.
            print("found a success image:%s compute:%d" % (file, compute_age))
        else:
            print("found a error image:%s compute:%d groud true:%d" % (file, compute_age, age))
        test_count += 1
    test_acc /= test_count
    test_age /= test_count
    print("{} Test validation{:.4f} mae:{:.4f} total_count:{}".format(datetime.now(), test_acc, test_age, test_count))

