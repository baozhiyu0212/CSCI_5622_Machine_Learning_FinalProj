import numpy as np
import os
import random
import sklearn

import tensorflow as tf
import re
import tensorflow.python.platform
from tensorflow.python.platform import gfile
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle

x_training = np.load('data/x_train.npy')
y_training = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')
label_dic = np.load('data/label_dic.npy')
list_images_training = np.load('data/list_images_training.npy')
list_images_test = np.load('data/list_images_test.npy')

model_dir = 'inception_v3'


def create_graph():
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            if ind % 100 == 0:
                print('Processing %s...' % image)
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
    return features


feat_training = extract_features(list_images_training)
feat_test = extract_features(list_images_test)

pickle.dump(feat_training, open('data/feat_training', 'wb'))
pickle.dump(feat_test, open('data/feat_test', 'wb'))


X_training = pickle.load(open('data/feat_training'))
X_test = pickle.load(open('data/feat_test'))


clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2')
clf.fit(X_training, y_training)
print clf.score(X_test, y_test)
