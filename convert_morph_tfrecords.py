"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import cv2
import numpy as np
RESIZE_WIDTH = 224
RESIZE_HEIGHT = 224

FLAGS = None
mean=np.array([129.1863, 104.7624, 93.5940])

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset_filename, name):
    if os.path.exists('tfrecords') == False:
        os.mkdir('tfrecords')
    """Converts a dataset to tfrecords."""
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for line in open(dataset_filename,'r'):
        line = line.replace('\n','')
        image_filename = line.split(' ')[0]
        label = line.split(' ')[1]
        image_raw = cv2.imread(image_filename)
        image_raw = cv2.resize(image_raw, (RESIZE_WIDTH, RESIZE_HEIGHT))
        #image_raw = image_raw.as_type(np.float32)
        #image_raw = image_raw.astype(np.float32)
        #image_raw -= mean
        #image_raw = image_raw.reshape((224*224*3))
        image_raw = image_raw.tostring()
        #image_raw_len = len(image_raw)
        #print("len is {}".format(image_raw_len))
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'images': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):
    # Convert to Examples and write the result to TFRecords.
    convert_to('morph_train.txt', 'train_%d_uint8' % RESIZE_HEIGHT)
    convert_to('morph_test.txt', 'test_%d_uint8' % RESIZE_HEIGHT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='./tfrecords',
        help='Directory to download data files and write the converted result'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
