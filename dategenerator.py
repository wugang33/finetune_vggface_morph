import numpy as np

import cv2

class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=False, mean = np.array([104., 117., 124.])
                 ,scale_size=(227,227), nb_classes=2):
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle
        self.nb_classes = nb_classes
        self.scale_size = scale_size
        self.class_list = class_list
        self.mean = mean
        self.pointer = 0

        self.read_class_list(class_list)

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.labels.append(int(items[1]))
            self.data_size = len(self.labels)

    def shuffle_data(self):
        images = self.images
        labels = self.labels
        self.images = []
        self.labels = []

        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        self.pointer = 0
        if self.shuffle:
            self.shuffle_data()


    def next_batch(self, batch_size):
        paths = self.images[self.pointer:self.pointer+batch_size]
        labels = self.labels[self.pointer: self.pointer+batch_size]
        self.pointer += batch_size

        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])
            if img is None:
                raise RuntimeError('can not read image file {}'.format(paths[i]))
            if self.horizontal_flip and np.random.random() <0.5:
                img = cv2.flip(img,1)

            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img = img.astype(np.float32)

            img -= self.mean
            images[i] = img
        one_hot_labels = np.zeros((batch_size,self.nb_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1
        return images, one_hot_labels

if __name__ == '__main__':
    train_file = './train.txt'
    val_file = './val.txt'
    train_generator = ImageDataGenerator(train_file,
                                         horizontal_flip = True, shuffle = True)
    val_generator = ImageDataGenerator(val_file, shuffle = False)
    for i in range(10):
        a = val_generator.next_batch(128)
        print(a)
