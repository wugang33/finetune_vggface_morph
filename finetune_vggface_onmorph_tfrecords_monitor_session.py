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
mean=np.array([129.1863, 104.7624, 93.5940])


train_layer = ['fc6', 'fc7', 'fc8']
display_step = 100
filewriter_path = "./finetune_vggface_morph/face"
checkpoint_path = "./finetune_vggface_morph/"

if os.path.isdir(checkpoint_path) is not True:
    os.mkdir(checkpoint_path)

def input(train = True):
    if train:
        filename = tf.train.string_input_producer(['./tfrecords/train_224_uint8.tfrecords'])
    else:
        filename = tf.train.string_input_producer(['./tfrecords/test_224_uint8.tfrecords'])
    reader = tf.TFRecordReader()
    key, serilized_example = reader.read(filename)
    features = tf.parse_single_example(serialized=serilized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'images': tf.FixedLenFeature([], tf.string)
    })
    age = tf.cast(features['label'], tf.int64)
    images = tf.decode_raw(features['images'], tf.uint8)
    images.set_shape([224*224*3])
    images = tf.reshape(images, [224, 224, 3])
    images = tf.cast(images, tf.float32)
    images = tf.subtract(images, mean)
    images = tf.image.random_flip_left_right(images)
#image = tf.reshape(features['images'], [224,224,3])

    image_batch, label_batch = \
        tf.train.shuffle_batch([images, age],
                               batch_size=batch_size, capacity=1000+batch_size*3,
                               num_threads=2,min_after_dequeue=1000)
    return image_batch, label_batch

image_batch,label_batch = input(True)
image_batch_test,label_batch_test = input(False)
is_test = tf.placeholder( dtype=tf.bool)
image_batch = tf.cond(is_test, lambda :image_batch_test, lambda :image_batch)
label_batch = tf.cond(is_test, lambda :label_batch_test, lambda :label_batch)

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
    train_op = optimizer.apply_gradients(gradients, global_step=tf.train.get_or_create_global_step())

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
saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_path, save_secs= 5*60)
writer = tf.summary.FileWriter(logdir=os.path.join(checkpoint_path,'face'))
merge_hook = tf.train.SummarySaverHook(save_secs=30,summary_writer=writer,
                                       output_dir=os.path.join(checkpoint_path,'face'),
                                       summary_op=tf.summary.merge_all())
model.load_initial_weights2()
init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

#init_ops2 = tf.group(tf.global_variables_initializer(),*init_ops)
#g_init_ops = tf.global_variables_initializer()

scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
with tf.train.SingularMonitoredSession(hooks=[saver_hook, merge_hook],
                                       checkpoint_dir=checkpoint_path, scaffold=scaffold) as sess:
    #for ops in g_init_ops:
    #sess.run(g_init_ops, feed_dict={is_test: False})
    #sess.run(init_ops2 ,feed_dict={is_test: False})
    for init_op in init_ops:
       init_val = sess.run(init_op, feed_dict={is_test: False})
       print(init_val)
    #sess.run(tf.global_variables_initializer())
    #model.load_initial_weights(sess)
    #saver.restore(sess, 'finetune_vggface_morph/model_epoch9.ckpk')
    writer.add_graph(sess.graph)
    print("{}Start train...".format(datetime.now()))
    print("{}Open tensorboard --logdir {}".format(datetime.now(), checkpoint_path))
    try:
        while not sess.should_stop():
            # Run training steps or whatever
            start = time.time()
            step,_ = sess.run([tf.train.get_or_create_global_step(), train_op], feed_dict={is_test:False})
            print('step:{} train_op consume {}'.format(step,time.time() - start))
            if step % 100 == 0 and step != 0:
                test_accuracy = sess.run(accuracy, feed_dict={is_test:True})
                print("test_accuracy is {}".format(test_accuracy))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    sess.close()