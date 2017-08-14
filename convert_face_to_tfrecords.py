"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import cv2
RESIZE_WIDTH = 40
RESIZE_HEIGHT = 40

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    if os.path.exists('tfrecords') == False:
        os.mkdir('tfrecords')
    """Converts a dataset to tfrecords."""
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for f, attrs in data_set.items():
        points = []
        f = "data80/%s.jpg"%(f)
        print(f)
        image_raw = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        image_raw = cv2.resize(image_raw, (RESIZE_WIDTH, RESIZE_HEIGHT))
        image_raw = image_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[f])),
            'gender': _int64_feature(attrs['gender']),
            'age': _int64_feature(attrs['age']),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def load_data(filename):
    train_data_sets = {}
    test_data_sets = {}
    id  = 0
    for line in open(filename).readlines():
        filename, gender,age = line.replace('\n','').split(',')
        gender = int(gender)
        age = int(age)
        #filename = 'data80/' + filename+'.jpg'
        id = id + 1
        if id % 10 == 0:
            test_data_sets[filename] = {'gender': gender, 'age': age}
        else:
            train_data_sets[filename] = {'gender': gender, 'age': age}
    return train_data_sets, test_data_sets


def main(unused_argv):
    train_data_sets, test_data_sets= load_data('datas.txt')

    # Convert to Examples and write the result to TFRecords.
    convert_to(train_data_sets, 'train_%d' % RESIZE_HEIGHT)
    convert_to(test_data_sets, 'test_%d' % RESIZE_HEIGHT)


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
