import tensorflow as tf
import os
from vgg_face_net import VggFace
import numpy as np
from dategenerator import ImageDataGenerator
train_file = './morph_train.txt'
val_file = './morph_test.txt'
from datetime import datetime
learn_rate = 0.0001
num_epochs = 30
batch_size = 32
import time

num_classes = 31


train_layer = ['fc6', 'fc7', 'fc8']
display_step = 100
filewriter_path = "./finetune_vggface_morph/face"
checkpoint_path = "./finetune_vggface_morph/"

if os.path.isdir(checkpoint_path) is not True:
    os.mkdir(checkpoint_path)

filename = tf.train.string_input_producer(['./tfrecords/train_224.tfrecords'])
reader = tf.TFRecordReader()
key, serilized_example = reader.read(filename)
features = tf.parse_single_example(serialized=serilized_example, features={
    'label': tf.FixedLenFeature([], tf.int64),
    'images': tf.FixedLenFeature([224,224,3], tf.float32)
})
age = tf.cast(features['label'], tf.int64)
images = features['images']

#image = tf.reshape(features['images'], [224,224,3])

image_batch, label_batch = \
    tf.train.shuffle_batch([images, age],
                           batch_size=batch_size, capacity=1000+batch_size*3,
                           num_threads=2,min_after_dequeue=1000)

tf.summary.image('input', image_batch)
model = VggFace(image_batch, num_classes, train_layer)

score = model.fc8

var_list = [var for var in tf.trainable_variables() if var.name.split('/')[0] in train_layer]

with tf.variable_scope('cross_entropy') as scope:
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=label_batch))

with tf.variable_scope('train') as scope:
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    train_op = optimizer.apply_gradients(gradients)

for gradient, var in gradients:
    tf.summary.histogram(var.name+'/gradients', gradient)

for var in var_list:
    tf.summary.histogram(var.name, var)

tf.summary.scalar('loss', loss)

with tf.variable_scope('accuracy') as scope:
    correct_pend = tf.equal(tf.argmax(score, 1), label_batch)
    accuracy = tf.reduce_mean(tf.cast(correct_pend, tf.float32))

tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()
#train_data_generator = ImageDataGenerator(train_file, horizontal_flip=True, shuffle=True,mean=np.array([129.1863, 104.7624, 93.5940]), scale_size=(224,224), nb_classes=num_classes)
#test_data_generator = ImageDataGenerator(train_file, horizontal_flip=False, shuffle=False,mean=np.array([129.1863, 104.7624, 93.5940]), scale_size=(224,224), nb_classes=num_classes)

#train_batches_per_epoch = np.floor(train_data_generator.data_size/batch_size).astype(np.int16)
#test_batches_per_epoch = np.floor(test_data_generator.data_size/batch_size).astype(np.int16)

writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)
    #saver.restore(sess, 'finetune_vggface_morph/model_epoch9.ckpk')
    writer.add_graph(sess.graph)
    print("{}Start train...".format(datetime.now()))
    print("{}Open tensorboard --logdir {}".format(datetime.now(), checkpoint_path))

    step = 0

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            start = time.time()
            sess.run(train_op)
            print('train_op consume {}'.format(time.time() - start))
            step = step + 1
            if step % display_step == 0 :
                start = time.time()
                s = sess.run(merged_summary)
                writer.add_summary(s, step)
                print('merge_summary consume {}'.format(time.time() - start))
            if step % 100 == 0 and step !=0:
                start = time.time()
                checkpoint_name = os.path.join(checkpoint_path, 'model_step' +str(step) + ".ckpk")
                saver.save(sess, checkpoint_name)
                print('saver consume {}'.format(time.time() - start))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    sess.close()