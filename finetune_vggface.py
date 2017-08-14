import tensorflow as tf
import os
from vgg_face_net import VggFace
import numpy as np
from dategenerator import ImageDataGenerator
train_file = './fgnet_train.txt'
val_file = './fgnet_val.txt'
from datetime import datetime
learn_rate = 0.0001
num_epochs = 100
batch_size = 32

num_classes = 100


train_layer = ['fc6', 'fc7', 'fc8']
display_step = 2
filewriter_path = "./finetune_vggface/face"
checkpoint_path = "./finetune_vggface/"

if os.path.isdir(checkpoint_path) is not True:
    os.mkdir(checkpoint_path)

X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
Y = tf.placeholder(dtype=tf.int64, shape=[None, num_classes])
#tf.summary.image('input', X)
model = VggFace(X, num_classes, train_layer)

score = model.fc8

var_list = [var for var in tf.trainable_variables() if var.name.split('/')[0] in train_layer]

with tf.variable_scope('cross_entropy') as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=Y))

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
    correct_pend = tf.equal(tf.argmax(score, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pend, tf.float32))

tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()
train_data_generator = ImageDataGenerator(train_file, horizontal_flip=True, shuffle=True,mean=np.array([129.1863, 104.7624, 93.5940]), scale_size=(224,224), nb_classes=num_classes)
test_data_generator = ImageDataGenerator(train_file, horizontal_flip=False, shuffle=False,mean=np.array([129.1863, 104.7624, 93.5940]), scale_size=(224,224), nb_classes=num_classes)

train_batches_per_epoch = np.floor(train_data_generator.data_size/batch_size).astype(np.int16)
test_batches_per_epoch = np.floor(test_data_generator.data_size/batch_size).astype(np.int16)

writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #model.load_initial_weights(sess)
    saver.restore(sess, 'finetune_vggface/model_epoch21.ckpk')
    writer.add_graph(sess.graph)
    print("{}Start train...".format(datetime.now()))
    print("{}Open tensorboard --logdir {}".format(datetime.now(), checkpoint_path))
    for epoch in range(22, num_epochs):
        for step in range(train_batches_per_epoch):
            x_xs,y_xs = train_data_generator.next_batch(batch_size)
            sess.run(train_op, feed_dict={X:x_xs, Y:y_xs})
            if step % display_step is 0:
                s = sess.run(merged_summary, feed_dict={X:x_xs, Y:y_xs} )
                writer.add_summary(s, epoch*train_batches_per_epoch+step)
        print("{}Start to validation...".format(datetime.now()))
        test_acc = 0
        test_count = 0
        for _ in range(test_batches_per_epoch):
            x_xs, y_xs = test_data_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={X:x_xs, Y:y_xs})
            test_acc+=acc
            test_count+=1
        test_acc/=test_count
        print("{}Test validation{:.4f}".format(datetime.now(), test_acc))
        print("{}Save checkpoint of model...".format(datetime.now()))
        train_data_generator.reset_pointer()
        test_data_generator.reset_pointer()
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+".ckpk")
        saver.save(sess,checkpoint_name)
        print("{}Chechpoint save at path {}".format(datetime.now(), checkpoint_name))
