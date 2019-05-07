# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:52:15 2019

@author: sonia
"""
from sklearn import svm

import pandas as pd
import numpy as np
import tensorflow as tf

import csv
import re
import os
import gc
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


from keras.models import model_from_json
from  keras.models import load_model
import numpy as np
import time
from keras import backend as K
from keras import optimizers
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib

#import seaborn as sns
import keract
import re
import json
import pandas as pd
from keras.utils import np_utils

import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator


import pandas as pd
import numpy as np

import csv
import re
import os

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.applications import VGG16
from keras_vggface.vggface import VGGFace
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
AveragePooling2D, Reshape, Permute, multiply, Dropout
from keras.callbacks import Callback
import numpy as np
import argparse
import csv
from keras import optimizers

from keras.models import model_from_json

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd
import keras.backend as K

import sys
import time, datetime, os
import matplotlib.pyplot as plt


import _pickle as pickle

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
#GLOBAL VAR

from keract import get_activations

import math


inputShape = (224, 224)
num_classes = 1
FREEZE = 7
BATCH_SIZE = 32
NB_EPOCHS = 15
NB_EPOCHS2 = 10
#-----------------------------------------------------------------------------#
# LOAD IMAGES

today =  datetime.date.today()
now= datetime.datetime.now()

todaystr = today.isoformat() + '-' + str(now.hour) + '-' + str(now.minute)
os.mkdir("results/" + todaystr)
log_file_path = "results/" + todaystr + "/log_file.txt"
sys.stdout = open(log_file_path, 'w')
os.mkdir("results/" + todaystr + "/all_logs_csv")
os.mkdir("results/" + todaystr + "/all_fc6_fc7")
os.mkdir("results/" + todaystr + "/all_svm_weights")


def load_images(tags_pict):
    """Load each image into list of numpy array and transform into array
    ----------
    dirname : path to folder with pictures
    tags_pict : pandas dataframe with annotation for pict
    Returns : array
    """
    img_data_list = []
    for p in tags_pict.index :
        # our input image is now represented as a NumPy array of shape
        # (inputShape[0], inputShape[1], 3) however we need to expand the
        # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
        # so we can pass it through thenetwork
        dirname = "D:\BDD_mandrillus_FACES_mars_2019"
        #img_path = tags_pict.folder[p] + '/' + tags_pict.Folder[p] + '/' + tags_pict.pict[p]
        img_path = dirname + "\\" +  tags_pict.pictfull[p]
        #print(img_path)
        img = load_img(img_path, target_size= inputShape)
        x = img_to_array(img)
        x = np.expand_dims(img, axis=0)
        # pre-process the image using the appropriate function based on the
        # model that has been loaded (i.e., mean subtraction, scaling, etc.)
        x = preprocess_input(x)
        img_data_list.append(x)
    img_data = np.array(img_data_list)
    img_data=np.rollaxis(img_data,1,0)
    img_data=img_data[0]
    print("End : load images")
    return(img_data)

# function to get unique values
def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    for x in unique_list:
        return(x,)


def euclidean_dist(vector1, vector2):
    '''calculate the euclidean distance
    input:  lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist


def calculate_rank(vector):
  a={}
  rank=1
  for num in sorted(vector):
    if num not in a:
      a[num]=rank
      rank=rank+1
  return[a[i] for i in vector]

class Histories(Callback):
    def on_train_begin(self,logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_accuracies.append(logs.get('val_acc'))


def plot_loss_acc_csv(hist_csv , todaystr):
    save_path = "results/" + todaystr + "/"
    train_loss=hist_csv['loss']
    #val_loss=hist_csv['val_loss']
    train_acc=hist_csv['acc']
    #val_acc=hist_csv['val_acc']
    xc=range(len(train_loss))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    #axes = plt.gca()
    #axes.set_ylim([0,1])
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.title('train_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.savefig(save_path + 'loss.png')
    plt.close()
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    #axes = plt.gca()
    #axes.set_ylim([0,1])
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.title('train_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.legend(['train'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig(save_path + 'acc.png')
    plt.close()

def split_get_act(model_custom, img_loaded, tags, n):
    img_loaded_split = np.array_split(img_loaded, n)
    for i in range(n):
        if i == 0 :
            fc6_all = get_activations(model_custom, img_loaded_split[i] , "fc6")['fc6/Relu:0']
            fc7_all = get_activations(model_custom, img_loaded_split[i] , "fc7")['fc7/Relu:0']
        else:
            fc6_all = np.concatenate((fc6_all,
                            get_activations(model_custom, img_loaded_split[i] , "fc6")['fc6/Relu:0']),
                            axis=0)
            fc7_all = np.concatenate((fc7_all,
                            get_activations(model_custom, img_loaded_split[i] , "fc7")['fc7/Relu:0']),
                            axis=0)
    fc6_all = pd.DataFrame(fc6_all)
    fc6_all.index = list(tags.index)
    fc7_all = pd.DataFrame(fc7_all)
    fc7_all.index = list(tags.index)
    return(fc6_all, fc7_all)


#import dataset
tags_pict = pd.read_csv('datas/pict_metadatas.csv')
#img_loaded = load_images(tags_pict)
tags_pict['sex'] = tags_pict['sex'].astype('category')
tags_pict['pred'] = len(tags_pict) * ["Nan"]

tags_pict['dist_centroid_fem_fc6'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_mal_fc6'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_fem_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_mal_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_mal_fc6_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['dist_centroid_fem_fc6_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['pred_fem_svm_fc6'] = len(tags_pict) * ["Nan"]
tags_pict['pred_fem_svm_fc7'] = len(tags_pict) * ["Nan"]
tags_pict['pred_fem_svm_fc6_fc7'] = len(tags_pict) * ["Nan"]


list_pict_comp = list(set(list(tags_pict.indiv[tags_pict.comportement == "yes"])))
list_pict_nocomp = list(set(list(tags_pict.indiv[tags_pict.comportement == "no"])))


#-----------------------------------------------------------------------------
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))


file_perf = open("results/" + todaystr + '/all_performances.txt', 'w')
#file_hist = open('all_histories.txt', 'w')

#len(list_pict_comp)
#for i in range(len(list_pict_comp)):
#for i in range(len(list_pict_nocomp) + len(list_pict_comp)):
#•for i in range(2):
L = sorted(list(set(list(tags_pict.indiv))))
print(L)

for i in range(len(L)):
#for i in range(3):
    print( "Tour :" + str(i) )
    #tags_train = tags_pict[-(tags_pict.indiv == list_pict_comp[i] )]
    #tags_test = tags_pict[tags_pict.indiv == list_pict_comp[i] ]
    tags_train = tags_pict[-(tags_pict.indiv == L[i] )]
    tags_test = tags_pict[tags_pict.indiv == L[i] ]
    print(tags_test)
    #make labels [fem,mal] and npy array
    label_train = np_utils.to_categorical(np.asarray(list(tags_train.sex.cat.codes), dtype ='int64'), len(tags_train.sex.cat.categories))
    label_test = np_utils.to_categorical(np.asarray(list(tags_test.sex.cat.codes), dtype ='int64'), len(tags_train.sex.cat.categories))
    img_loaded_train = load_images(tags_train)
    img_loaded_test = load_images(tags_test)
    #  MODELS  vggface
    model = VGGFace(model='vgg16',include_top=False, input_shape=(224, 224, 3))
    last_layer = model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    #out = Dense(y_train.shape[1], activation='softmax', name='fc8/pred')(x)
    #out = Dense(num_classes, activation='softmax', name='fc8/pred')(x)
    out = Dense(num_classes, activation='sigmoid', name='fc8/pred')(x)
    model_custom = Model( input=model.input, outputs = out)
    print("Architecture custom")
    print(model_custom.summary())
    for layer in model_custom.layers[:FREEZE]:
        layer.trainable = False
    for layer in model_custom.layers:
        print(layer, layer.trainable)
    #model_custom.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001), metrics=['accuracy'] )
    #sgd2 = optimizers.SGD(lr=0.0001,  momentum=0.9, decay=0.00001, nesterov=True)
    adam2 = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
    #model_custom.compile(optimizer=sgd2, loss='binary_crossentropy', metrics=['accuracy'])
    model_custom.compile(optimizer=adam2, loss='binary_crossentropy', metrics=['accuracy'])
    #Save learning steps and callbacks
    histories = Histories()
    histories2 = Histories()
    filelog = "log_" + str(L[i]) + ".csv"
    callbacks=[histories,
               CSVLogger("results/" + todaystr + "/all_logs_csv/"  + filelog , append=True, separator=';')  ,
               #EarlyStopping(patience=10),
               #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,verbose = 1,cooldown=1)
               ]
    callbacks2=[histories2,
               CSVLogger("results/" + todaystr + "/all_logs_csv/"  + filelog , append=True, separator=';')  ,
               #EarlyStopping(patience=10),
               #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,verbose = 1,cooldown=1)
               ]
    #x,y = shuffle(img_loaded_train, list(tags_train.sex.cat.codes))
    #X_learn, X_val, y_learn, y_val = train_test_split(x,y,test_size=0.2)
    X_learn = img_loaded_train
    y_learn = tags_train.sex.cat.codes
    y_learn_tags = tags_train
    #y_learn_tags = tags_train
    # Create train generator.
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=30,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2,
                                       horizontal_flip  =True)
    train_generator = train_datagen.flow(X_learn, y_learn, shuffle=False,
                                         batch_size=BATCH_SIZE, seed=1)
    # Create validation generator
    val_datagen = ImageDataGenerator(rescale = 1./255)
    #val_generator = val_datagen.flow(X_val, y_val, shuffle=False, batch_size=BATCH_SIZE, seed=1)
    train_steps_per_epoch = X_learn.shape[0] //  BATCH_SIZE
    #val_steps_per_epoch = X_val.shape[0] //  BATCH_SIZE
    hist = model_custom.fit_generator(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  #validation_data=val_generator,
                                  #validation_steps=val_steps_per_epoch,
                                  epochs=NB_EPOCHS, verbose=1,
                                  callbacks=callbacks)
    sgd2 = optimizers.SGD(lr=0.00001,  momentum=0.9, decay=0.0, nesterov=True)
    #sgd2 = optimizers.SGD(lr=0.0001,  momentum=0.9, decay=0.0, nesterov=True)
    model_custom.compile(optimizer=sgd2, loss='binary_crossentropy', metrics=['accuracy'])
    hist2 =  model_custom.fit_generator(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  #validation_data=val_generator,
                                  #validation_steps=val_steps_per_epoch,
                                  epochs=NB_EPOCHS2, verbose=1,
                                  callbacks=callbacks2)
    #scorebis = model_custom.evaluate(img_loaded_test, list(tags_test.sex.cat.codes))
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_generator = test_datagen.flow(img_loaded_test, list(tags_test.sex.cat.codes), shuffle=False, batch_size=1, seed=1)
    scorebis = model_custom.evaluate_generator(test_generator,steps=(test_generator.n))
    #print('Final test accuracy:', (scorebis[1]*100.0))
    print('Final test accuracy for '+ L[i] + ":" +  str(scorebis[1]*100.0) )
    file_perf.write('Final test accuracy for '+ L[i] + " : " +  str(scorebis[1]*100.0) + "\n")
    #val_generator.reset()
    pred = model_custom.predict_generator(test_generator,verbose=1, steps= test_generator.n )
    print(pred)
    tags_test['pred'] = [item for sublist in pred.tolist() for item in sublist]
    print(tags_test['pred'])
    #ACTIVATIONS TEST FC6 et FC7
    fc6_fc7_splits = split_get_act(model_custom, img_loaded_test, tags_test, 3)
    all_fc6_df = fc6_fc7_splits[0]
    all_fc7_df = fc6_fc7_splits[1]
    #ACTIVATIONS TRAIN FC6 et FC7
    fc6_fc7_splits = split_get_act(model_custom, X_learn, y_learn_tags, 20)
    fc6_train_all = fc6_fc7_splits[0]
    fc7_train_all = fc6_fc7_splits[1]
    print(len(fc6_train_all))
    print(len(fc7_train_all))
    fc6_fc7 = pd.concat([all_fc6_df, all_fc7_df], axis=1)
    train_fc6_fc7 = pd.concat([fc6_train_all, fc7_train_all], axis=1)
    print(train_fc6_fc7)
    list_train_fem = list(tags_train.index[tags_train.sex == "fem"])
    list_train_mal = list(tags_train.index[tags_train.sex == "mal"])
    fc6_train_fem = fc6_train_all.ix[list_train_fem]
    fc6_train_mal = fc6_train_all.ix[list_train_mal]
    fc7_train_fem = fc7_train_all.ix[list_train_fem]
    fc7_train_mal = fc7_train_all.ix[list_train_mal]
    fc6_fc7_train_fem = train_fc6_fc7.ix[list_train_fem]
    fc6_fc7_train_mal = train_fc6_fc7.ix[list_train_mal]
    fc6_fc7 = pd.concat([all_fc6_df, all_fc7_df], axis=1)
    train_fc6_fc7 = pd.concat([fc6_train_all, fc7_train_all], axis=1)
    centroide_fem_fc6 = list(fc6_train_fem.mean(axis = 0))
    centroide_mal_fc6 = list(fc6_train_mal.mean(axis = 0))
    centroide_fem_fc7 = list(fc7_train_fem.mean(axis = 0))
    centroide_mal_fc7 = list(fc7_train_mal.mean(axis = 0))
    centroide_fem_fc6_fc7 = list(fc6_fc7_train_fem.mean(axis = 0))
    centroide_mal_fc6_fc7 = list(fc6_fc7_train_mal.mean(axis = 0))
    tags_test['dist_centroid_fem_fc6'] = list(all_fc6_df.apply(lambda row: euclidean_dist(row,centroide_fem_fc6 ), axis=1))
    tags_test['dist_centroid_mal_fc6'] = list(all_fc6_df.apply(lambda row: euclidean_dist(row,centroide_mal_fc6 ), axis=1))
    tags_test['dist_centroid_fem_fc7'] = list(all_fc7_df.apply(lambda row: euclidean_dist(row,centroide_fem_fc7 ), axis=1))
    tags_test['dist_centroid_mal_fc7'] = list(all_fc7_df.apply(lambda row: euclidean_dist(row,centroide_mal_fc7 ), axis=1))
    tags_test['dist_centroid_mal_fc6_fc7'] = list(fc6_fc7.apply(lambda row: euclidean_dist(row,centroide_mal_fc6_fc7 ), axis=1))
    tags_test['dist_centroid_fem_fc6_fc7'] = list(fc6_fc7.apply(lambda row: euclidean_dist(row,centroide_fem_fc6_fc7 ), axis=1))
    classifier = svm.SVC(kernel='linear', C=0.01, probability=True)
    classifier.fit(fc6_train_all, list(tags_train.sex.cat.codes))
    pred_prob_svm = classifier.predict_proba(all_fc6_df)
    weights1 = classifier.coef_
    classifier = svm.SVC(kernel='linear', C=0.01, probability=True)
    classifier.fit(fc7_train_all, list(tags_train.sex.cat.codes))
    pred_prob_svm_2 = classifier.predict_proba(all_fc7_df)
    weights2 = classifier.coef_
    classifier = svm.SVC(kernel='linear', C=0.01, probability=True)
    classifier.fit(train_fc6_fc7, list(tags_train.sex.cat.codes))
    pred_prob_svm_3 = classifier.predict_proba(fc6_fc7)
    weights3 = classifier.coef_
    #hist_csv = pd.read_csv("results/" + todaystr + "/all_logs_csv/"  + filelog  , sep=';' )
    tags_test['pred_fem_svm_fc6'] = list(pred_prob_svm[:,0])
    tags_test['pred_fem_svm_fc7'] = list(pred_prob_svm_2[:,0])
    tags_test['pred_fem_svm_fc6_fc7'] = list(pred_prob_svm_3[:,0])
    for img in list(tags_test.pict):
        print(tags_pict.pred[tags_pict.pict == img])
        print(tags_test[tags_test.pict == img])
        tags_pict.pred[tags_pict.pict == img] = tags_test.pred[tags_test.pict == img]
        tags_pict['dist_centroid_fem_fc6'][tags_pict.pict == img] = tags_test['dist_centroid_fem_fc6'][tags_test.pict == img]
        tags_pict['dist_centroid_mal_fc6'][tags_pict.pict == img] = tags_test['dist_centroid_mal_fc6'][tags_test.pict == img]
        tags_pict['dist_centroid_fem_fc7'][tags_pict.pict == img] = tags_test['dist_centroid_fem_fc7'][tags_test.pict == img]
        tags_pict['dist_centroid_mal_fc7'][tags_pict.pict == img] = tags_test['dist_centroid_mal_fc7'][tags_test.pict == img]
        tags_pict['dist_centroid_fem_fc6_fc7'][tags_pict.pict == img] = tags_test['dist_centroid_fem_fc6_fc7'][tags_test.pict == img]
        tags_pict['dist_centroid_mal_fc6_fc7'][tags_pict.pict == img] = tags_test['dist_centroid_mal_fc6_fc7'][tags_test.pict == img]
        tags_pict['pred_fem_svm_fc6'][tags_pict.pict == img] = tags_test['pred_fem_svm_fc6'][tags_test.pict == img]
        tags_pict['pred_fem_svm_fc7'][tags_pict.pict == img] = tags_test['pred_fem_svm_fc7'][tags_test.pict == img][tags_test.pict == img]
        tags_pict['pred_fem_svm_fc6_fc7'][tags_pict.pict == img] = tags_test['pred_fem_svm_fc6_fc7'][tags_test.pict == img]
    all_w = pd.DataFrame([weights1.tolist(), weights2.tolist(), weights3.tolist()])
    all_w.to_csv("results/" + todaystr + "/all_svm_weights/" + filelog)
    fc_to_save = pd.concat([fc6_fc7, train_fc6_fc7])
    fc_to_save.to_csv("results/" + todaystr + "/all_fc6_fc7/" + filelog)
    reset_keras()
    tags_pict.to_csv("results/" + todaystr + "/tags_pict_pred_tmp.csv")


file_perf.close()

tags_pict.to_csv("results/" + todaystr + "/tags_pict_pred.csv")


sys.stdout.close()
