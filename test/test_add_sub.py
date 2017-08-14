import tensorflow as tf

a = tf.get_variable('test',dtype=tf.int32, initializer=0)

b = tf.assign(a,10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(b)
    print(sess.run(a))

sess.close()

with tf.Session() as sess1:
    print(sess1.run(a))