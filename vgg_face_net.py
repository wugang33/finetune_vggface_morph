import tensorflow as tf
import numpy as np
import os

class VggFace:
    def __init__(self, X, num_classes, skip_layers, weights_path='DEFAULT'):
        self.X = X
        self.num_classes = num_classes
        self.skip_layers = skip_layers
        self.weights_path = weights_path
        if weights_path == 'DEFAULT':
            self.weights_path = os.path.join('vggface','vggface_model.npy')
        self.create()

        pass

    def create(self):
        conv1_1 = self.conv(self.X, 3,3,64,1,1,'conv1_1')
        conv1_2 = self.conv(conv1_1, 3, 3, 64, 1, 1, 'conv1_2')
        pool1 = self.max_pool(conv1_2, 2,2,2,2,'pool1')
        conv2_1 = self.conv(pool1, 3,3,128,1,1,'conv2_1')
        conv2_2 = self.conv(conv2_1,3,3,128, 1,1,'conv2_2')
        pool2 = self.max_pool(conv2_2, 2,2,2,2,'pool2')
        conv3_1 = self.conv(pool2, 3,3,256,1,1,'conv3_1')
        conv3_2 = self.conv(conv3_1,3,3,256,1,1,'conv3_2')
        conv3_3 = self.conv(conv3_2,3,3,256,1,1,'conv3_3')
        pool3 = self.max_pool(conv3_3, 2,2,2,2,'pool3')
        conv4_1 = self.conv(pool3, 3,3,512,1,1,'conv4_1')
        conv4_2 = self.conv(conv4_1,3,3,512,1,1,'conv4_2')
        conv4_3 = self.conv(conv4_2,3,3,512,1,1,'conv4_3')
        pool4 = self.max_pool(conv4_3, 2,2,2,2,'pool4')
        conv5_1 = self.conv(pool4, 3,3,512,1,1,'conv5_1')
        conv5_2 = self.conv(conv5_1, 3,3,512,1,1,'conv5_2')
        conv5_3 = self.conv(conv5_2, 3,3,512,1,1,'conv5_3')
        pool5 = self.max_pool(conv5_3, 2,2,2,2,'pool5')
        pool5_shape = pool5.get_shape().as_list()
        pool5_reshape = tf.reshape(pool5, [-1, pool5_shape[1]*pool5_shape[2]*pool5_shape[3]])
        fc6 = self.fc(pool5_reshape, 4096, 'fc6')
        fc7 = self.fc(fc6, 4096,  'fc7')
        self.fc7 = fc7
        self.fc8 = self.fc(fc7, self.num_classes,need_relu=False, name='fc8')

    def load_initial_weights2(self):
        weights = np.load(self.weights_path, encoding='bytes')
        weights = weights.item()
        for op_name in weights:
            print(op_name)
            if op_name in self.skip_layers:
                continue
            with tf.variable_scope(op_name, reuse=True) as scope:
                for key in weights[op_name]:
                    print(key)
                    data = weights[op_name][key]
                    a = tf.assign(tf.get_variable(key), data)
                    tf.add_to_collection(tf.GraphKeys.INIT_OP, a)

    def load_initial_weights(self, session):
        weights = np.load(self.weights_path, encoding='bytes')
        weights = weights.item()
        for op_name in weights:
            print(op_name)
            if op_name in self.skip_layers:
                continue
            with tf.variable_scope(op_name, reuse=True) as scope:
                for key in weights[op_name]:
                    print(key)
                    data = weights[op_name][key]
                    session.run(tf.assign(tf.get_variable(key), data))


    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights',
                                      shape=[filter_height, filter_width, input_channels, num_filters ])
            biases = tf.get_variable('biases', shape=[num_filters])

            conv = tf.nn.conv2d(x, weights, [1, stride_y, stride_x, 1], padding)
            act =  tf.nn.relu(tf.nn.bias_add(conv, biases))
        return act

    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding="SAME"):
        return tf.nn.max_pool(x, [1, filter_height, filter_width, 1], [1, stride_y, stride_x, 1], padding=padding,name=name)

    def fc(self, x, num_outputs,name, need_relu = True):
        num_inputs = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_inputs, num_outputs])
            biases = tf.get_variable('biases', shape=[num_outputs])
            tc = tf.nn.xw_plus_b(x, weights, biases)
            if need_relu:
                tc = tf.nn.relu(tc)
            return tc

if __name__== '__main__':
    img = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
    net = VggFace(img, 2622 , [], 'DEFAULT')
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    net.load_initial_weights(session)