import os
import cv2
import re
import numpy as np

import matplotlib.pyplot as plt

def generate_doc_and_cat():
    trainfile = open('train.txt','w')
    valfile = open('val.txt','w')

    for i in range(500):
        cat_filename = os.path.join('data','train','cat.%d.jpg' % i)
        dog_filename = os.path.join('data','train','dog.%d.jpg' % i)
        trainfile.write('%s 1\n'%dog_filename)
        trainfile.write('%s 0\n' % cat_filename)
    for i in range(500,700):
        cat_filename = os.path.join('data','train','cat.%d.jpg' % i)
        dog_filename = os.path.join('data','train','dog.%d.jpg' % i)
        valfile.write('%s 1\n'%dog_filename)
        valfile.write('%s 0\n' % cat_filename)


def generate_AR():
    trainfile = open('fgnet_train.txt' ,'w')
    testfile = open('fgnet_val.txt', 'w')
    for id in range(1,51):
        random_choose = np.random.permutation(26)+1
        for i, num in enumerate(random_choose):
            filename = os.path.join('data', 'AR', 'm-%03d-%02d.pgm' % (id, num))
            if i<3:
                testfile.write('%s %d\n' % (filename, id-1))
            else:
                trainfile.write('%s %d\n' % (filename, id-1))
        random_choose = np.random.permutation(26) + 1
        for i, num in enumerate(random_choose):
            filename = os.path.join('data', 'AR', 'w-%03d-%02d.pgm' % (id, num))
            if i < 3:
                testfile.write('%s %d\n' % (filename, id+49))
            else:
                trainfile.write('%s %d\n' % (filename, id+49))


def generate_fg_net():
    r = r'(\d\d\d)[Aa](\d\d)(a|b)*\.'
    regex = re.compile(r)
    path = '/home/wg/data/fgnet_finetune/FG-NET/Images'
    images = os.listdir(path)
    train_file = open('fgnet_age_train.txt', 'w')
    valid_file = open('fgnet_age_val.txt', 'w')
    age_analysis = open('age_analysis.txt','w')
    idx = 0
    ages = []
    plt.figure(figsize=(10,6))
    for image in images:
        idx += 1
        print(image)
        id, age, _ = regex.findall(image)[0]
        print(age)
        age_analysis.write('%s\n'%(age))
        ages.append(int(age))
        #train_file.write('%s %s\n' % (image, age))
        #if idx % 3 is 0:
        #valid_file.write('%s %s\n' % (image, age))
    plt.hist(ages, bins = np.max(ages) - np.min(ages), normed=1 , facecolor='blue' , alpha=0.5)
    plt.waitforbuttonpress()


def analysis_morph():
    path = '/home/wg/data/morph/MORPH'
    images = os.listdir(path)
    r = r'(\d+)_(\d+)(M|F)(\d+).JPG'
    regex = re.compile(r)
    ages = []
    for image in images:
        print(image)
        id,seq,gender,age = regex.findall(image)[0]
        ages.append(int(age))
    plt.figure(figsize=(15,6))
    plt.hist(ages, bins=np.max(ages) - np.min(ages), normed=False, facecolor='blue', alpha=0.5)
    plt.waitforbuttonpress()

def generate_morph_age_dataset():
    datapath = '/home/wg/data/morph/MORPH'
    files = os.listdir(datapath)
    r = r'(\d+)_(\d+)(M|F)(\d+).JPG'
    regex = re.compile(r)
    ages = []
    r = r''
    for age in range(16,47):
        ages.append(([], []))
    for file in files:
        id, seq, gender, age = regex.findall(file)[0]
        age = int(age)
        ages_index = age-16
        if age<16 or age > 46:
            continue
        if gender is 'M':
            ages[ages_index][0].append(file)
        elif gender is 'F':
            ages[ages_index][1].append(file)
        else:
            raise RuntimeError('error gender for file {}'.format(file))
    choose_file = open('morph_choose_train.txt', 'w')
    choose_test_file = open('morph_choose_test.txt', 'w')
    for age,files in enumerate(ages):
        age = age + 16
        print('current age is :{}'.format(age))
        m_files = files[0]
        f_files = files[1]
        assert len(m_files) >= 110
        assert len(f_files) >= 110
        m_idx = np.random.permutation(len(m_files))
        f_idx = np.random.permutation(len(f_files))
        for i in range(0, 100):
            choose_m = m_files[m_idx[i]]
            choose_f = f_files[f_idx[i]]
            choose_file.write('%s\n' % choose_m)
            choose_file.write('%s\n' % choose_f)
        for i in range(100, 110):
            choose_m = m_files[m_idx[i]]
            choose_f = f_files[f_idx[i]]
            choose_test_file.write('%s\n' % choose_m)
            choose_test_file.write('%s\n' % choose_f)

import dlib
detector = dlib.get_frontal_face_detector()
def process_morph_database(txtname):
    lines = open(txtname).readlines()
    datapath = '/home/wg/data/morph/MORPH'
    cropedpath = '/home/wg/data/morph/MORPH_croped'
    if os.path.isdir(cropedpath) is not True:
        os.mkdir(cropedpath)
    for line in lines:
        imgname = line.replace('\n','')
        img = cv2.imread(os.path.join(datapath, imgname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(img)
        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
            if d.top() < 0 or d.left() <0 or d.right() <0 or d.bottom() <0:
                continue
            if d.bottom() >= img.shape[0] or d.right() >= img.shape[1]:
                continue
            img_croped = img[d.top():d.bottom(), d.left():d.right(), :]
            img_croped = cv2.resize(img_croped,(224,224))
            img_croped = cv2.cvtColor(img_croped, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(cropedpath, imgname), img_croped)

def generate_label_for_morph():
    cropedpath = '/home/wg/data/morph/MORPH_croped'
    train = open('morph_train.txt','w')
    test = open('morph_test.txt','w')
    for line in open('morph_choose_test.txt'):
        line = line.replace('\n','')
        if os.path.exists(os.path.join(cropedpath, line)) is False:
            continue
        r = r'(\d+)_(\d+)(M|F)(\d+).JPG'
        regex = re.compile(r)
        id, seq, gender, age = regex.findall(line)[0]
        age = int(age)
        test.write('%s %d\n'%(os.path.join(cropedpath, line), age-16))
    for line in open('morph_choose_train.txt'):
        line = line.replace('\n', '')
        if os.path.exists(os.path.join(cropedpath, line)) is False:
            continue
        r = r'(\d+)_(\d+)(M|F)(\d+).JPG'
        regex = re.compile(r)
        id, seq, gender, age = regex.findall(line)[0]
        age = int(age)
        train.write('%s %d\n'%(os.path.join(cropedpath, line),age-16))

if __name__ == "__main__":
    #generate_fg_net()
    #analysis_morph()
    #generate_AR()
    #generate_morph_age_dataset()
    #process_morph_database('morph_choose_train.txt')
    generate_label_for_morph()