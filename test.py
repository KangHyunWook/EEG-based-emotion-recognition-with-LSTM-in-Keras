#Date: 2021-7-30
#Programmer: HYUN WOOK KANG

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout    

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import f1_score

import numpy as np
import os
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('-w', '--weights', help='name of the saved weights')
args=ap.parse_args()
args=vars(args)

import pickle

with open('test_data.pkl', 'rb') as f:
    test_data=pickle.load(f)

data=test_data[:,:-1]
labels=test_data[:,-1]

le = LabelEncoder()
le.fit(labels)
labels=le.transform(labels)

testX=data=data.reshape(data.shape+(1,))


"""Pass the train data to the model"""
# model.compile(loss='categorical_crossentropy', optimizer='adam')
saved_model_path=os.path.join('./model', args['weights'])
model = load_model(saved_model_path)
"""Load saved model weights"""
import os 

print('====Performance====')

predictions=model.predict(testX, verbose=0)
corr=0


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
y_true=[]
y_pred=[]

for i in range(len(predictions)):
    idx=np.argmax(predictions[i])
    y_pred.append(idx)
    y_true.append(labels[i])
    if(idx==labels[i]):
        corr+=1

print('acc: {0:.4f}'.format(corr/len(predictions)))
print('f1 score: {0:.4f}'.format(f1_score(y_true, y_pred, average='weighted')))





