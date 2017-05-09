import cv2
import numpy as np
import os
import random
from PIL import Image
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from scipy.misc import imsave

label_vol = {}
x_train = []
y_train = []
x_test = []
y_test = []


def fix_size(final_path, width, height):
    img = Image.open(final_path)
    out = img.resize((width, height), Image.ANTIALIAS)
    return out


def Labelexp(label_vol, num, label):
    if num not in label_vol:
        label_vol[num] = label
    return label_vol


def DataInitialize():
    cat = 0
    list_images_training = []
    list_images_test = []
    for filename in os.listdir("Caltech101"):
        if filename == ".DS_Store":
            continue
        cat += 1
        # num-label dictionary
        label_dic = Labelexp(label_vol, cat, filename)

        current_path = "Caltech101/" + filename
        files = os.listdir(current_path)
        random.shuffle(files)
        print "splitting category :" + filename
        Num_train = 20
        Num_test = min(len(os.listdir(current_path)) - Num_train, 30)
        index = [i for i in range(Num_test + Num_train)]
        random.shuffle(index)
        seed_train = [index[i] for i in range(Num_train)]
        cnt = 0
        for pic in files:
            final_path = current_path + "/" + pic
            fixed_img = fix_size(final_path, 300, 200)
            gray_img = fixed_img.convert('L')
            final_img = np.array(gray_img).ravel()

            if cnt < Num_test + Num_train:
                if cnt not in seed_train:
                    x_test.append(final_img)
                    y_test.append(cat)
                    list_images_test.append(final_path)
                else:
                    x_train.append(final_img)
                    y_train.append(cat)
                    list_images_training.append(final_path)
            else:
                break
            cnt += 1
    np.save('data/x_train.npy', x_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/x_test.npy', x_test)
    np.save('data/y_test.npy', y_test)
    np.save('data/label_dic.npy', label_dic)
    np.save('data/list_images_training', list_images_training)
    np.save('data/list_images_test', list_images_test)
    return x_train, y_train, x_test, y_test, label_dic


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, label_dic = DataInitialize()
    # x_train = np.load('data/x_train.npy')
    # y_train = np.load('data/y_train.npy')
    # x_test = np.load('data/x_test.npy')
    # y_test = np.load('data/y_test.npy')
    # label_dic = np.load('data/label_dic.npy')



    # nei = KNeighborsClassifier(n_neighbors=3)
    # nei.fit(x_train, y_train)
    # print nei.score(x_test, y_test)

    # tree = DecisionTreeClassifier()
    # tree.fit(x_train, y_train)
    # print tree.score(x_test, y_test)

    # lr = LogisticRegression(multi_class='multinomial', solver='sag', verbose=10, tol=1e-3)
    # lr.fit(x_train, y_train)
    # print('Training Complete')
    # print lr.score(x_test, y_test)
