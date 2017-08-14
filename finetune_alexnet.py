import tensorflow as tf
import numpy as np
from datetime import datetime
from alexnet import AlexNet
import os
import time
from dategenerator import ImageDataGenerator
train_file = './train.txt'
val_file = './val.txt'

learn_rate = 0.001
num_epochs = 10
batch_size = 128
drop_rate = 0.5
num_classes = 2
train_layer = ['fc7', 'fc8']
display_step = 1
filewriter_path = "./finetune_alexnet/dogs_vs_cats"
checkpoint_path = "./finetune_alexnet/"

if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

x = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3])
y = tf.placeholder(dtype=tf.int64, shape=[None, num_classes])
keep_prob = tf.placeholder(dtype=tf.float32)

model = AlexNet(x, keep_prob, num_classes, train_layer)
score = model.fc8

var_list = [var for var in tf.trainable_variables() if var.name.split('/')[0] in train_layer]

with tf.name_scope('corss_ent'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

with tf.name_scope('train'):
    '''
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    train_op = optimizer.apply_gradients(gradients)
    '''
    train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, var_list=var_list)
'''
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)
'''
for var in var_list:
    tf.summary.histogram(var.name, var)

tf.summary.scalar('cross_entropy', loss)


with tf.name_scope('accuracy'):
    correct_pend = tf.equal(tf.argmax(score,1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pend, tf.float32))

tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()
train_generator = ImageDataGenerator(train_file,
                                     horizontal_flip = True, shuffle = True)
val_generator = ImageDataGenerator(val_file, shuffle = False)
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)
    #saver.restore(sess, "finetune_alexnet/model_epoch9.ckpk")
    writer.add_graph(sess.graph)
    print("{} start train...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), checkpoint_path))
    for epoch in range(num_epochs):
        print("{} Epochs Number: {}".format(datetime.now(), epoch + 1))
        step = 1
        start= time.time()
        while step < train_batches_per_epoch:
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob:drop_rate})
            if step % display_step ==0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
                writer.add_summary(s,  epoch*train_batches_per_epoch + step)
            step += 1
        print("{} Each step consume time:{}".format(datetime.now(), (time.time() - start)/train_batches_per_epoch))
        print("{} Start Validation".format(datetime.now()))
        test_acc = 0
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_xs, batch_ys = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Test Validation {:.4f}".format(datetime.now(), test_acc))
        print("{} Save checkpoint of model...".format(datetime.now()))
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpk')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Checkpoint save at path: {}".format(datetime.now(), checkpoint_name))