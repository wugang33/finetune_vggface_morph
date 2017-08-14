import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from tensorflow.contrib.slim.nets import alexnet
class AlexNet(object):

    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

        pass

    def create(self):
        '''
          def setup(self):
                (self.feed('data')
                     .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
                     .lrn(2, 1.99999994948e-05, 0.75, name='norm1')
                     .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
                     .conv(5, 5, 256, 1, 1, group=2, name='conv2')
                     .lrn(2, 1.99999994948e-05, 0.75, name='norm2')
                     .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
                     .conv(3, 3, 384, 1, 1, name='conv3')
                     .conv(3, 3, 384, 1, 1, group=2, name='conv4')
                     .conv(3, 3, 256, 1, 1, group=2, name='conv5')
                     .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
                     .fc(4096, name='fc6')
                     .fc(4096, name='fc7')
                     .fc(1000, relu=False, name='fc8')
        '''
        conv1 = self.conv(self.X, 11, 11, 96, 4, 4, name='conv1', padding='VALID')
        lrn1 = self.lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = self.max_pool(lrn1, 3, 3, 2, 2, padding='VALID', name='pool1')
        conv2 = self.conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        lrn2 = self.lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = self.max_pool(lrn2, 3, 3, 2, 2, padding='VALID', name='pool2')
        conv3 = self.conv(pool2, 3, 3, 384, 1, 1, name='conv3')
        conv4 = self.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        conv5 = self.conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = self.fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = self.dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = self.fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = self.dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = self.fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

        pass

    def load_initial_weights(self, session):
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()
        '''is a dict of dcit {'conv1':{'weights':array,'biases':array}'''
        for op_name in weights_dict:
            print(op_name)
            if op_name in self.SKIP_LAYER:
               continue
            with tf.variable_scope(op_name, reuse=True) as scope:
                for key in weights_dict[op_name]:
                    print(key)
                    var = tf.get_variable(key)#, trainable=False)
                    session.run(tf.assign(var, weights_dict[op_name][key]))


        pass

    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
        input_channels = int(x.get_shape()[-1])
        convolve = lambda i,k : tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
            if groups == 1:
                conv = convolve(x, weights)
            else:
                inputs_groups = tf.split(value=x, num_or_size_splits=groups, axis=3)
                weights_groups = tf.split(value=weights, num_or_size_splits=groups, axis=3)
                output_groups = [convolve(i,k) for i, k in zip(inputs_groups, weights_groups)]
                conv = tf.concat(output_groups, axis=3)
            bias = tf.nn.bias_add(value=conv, bias=biases)
            #bias = tf.reshape(tf.nn.bias_add(value=conv, bias=biases), shape=conv.get_shape().as_list())
            act = tf.nn.relu(bias, name=scope.name)
        return act

    def fc(self, x, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in,num_out], trainable=True)
            biases = tf.get_variable('biases', shape=[num_out])
            act = tf.nn.xw_plus_b(x, weights, biases=biases, name=scope.name)
        if relu is True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name,
                 padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, filter_height,filter_width, 1], strides=[1, stride_y, stride_x,1], padding=padding, name=name)

    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        """Create a local response normalization layer."""
        return tf.nn.local_response_normalization(x, depth_radius=radius,
                                                  alpha=alpha, beta=beta,
                                                  bias=bias, name=name)

    def dropout(self,x, keep_prob):
        """Create a dropout layer."""
        return tf.nn.dropout(x, keep_prob)

if __name__ == '__main__':
    img = tf.placeholder(tf.float32, shape=[1, 227,227,3])
    net = AlexNet(img, 0.5, 1000, [],'DEFAULT')
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    net.load_initial_weights(session)