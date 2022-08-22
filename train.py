#Date: 2021-7-30
#Programmer: HYUN WOOK KANG
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout    
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import load_model

import numpy as np
import os
import pandas as pd
import argparse

def aug_white_noise(X, rate):
    
    #white noise
    noise_X=[]
    n_num=int(len(X)*rate)
    for i in range(n_num):

        noise_amp=0.005*np.random.uniform()*np.amax(X[i])
        noise_x=X[i].astype('float64')+noise_amp*np.random.normal(size=X[i].shape)
        noise_X.append(noise_x)

    noise_X=np.array(noise_X)
    
    return noise_X

ap = argparse.ArgumentParser()
ap.add_argument('--vis', action='store_true')
ap.add_argument('--weights')
ap.add_argument('--aug', action='store_true')
args = ap.parse_args()
args=vars(args)

df_eeg_data=pd.read_csv('emotions.csv')

df_eeg_data=shuffle(df_eeg_data, random_state=7)

data=df_eeg_data.iloc[:,:-1]
data=data.values
labels=df_eeg_data.iloc[:,-1]
labels=labels.values

le = LabelEncoder()
le.fit(labels)
labels=le.transform(labels)

n_val_data=int(len(data)*0.2)
n_test_data=int(len(data)*0.2)
n_train_data=len(data)-n_val_data-n_test_data

train_data=data[:n_train_data]
train_labels=labels[:n_train_data]

val_data=data[n_train_data:n_train_data+n_val_data]
val_labels=labels[n_train_data:n_train_data+n_val_data]

print(train_data.shape)

if args['aug']:
    noise_train_data=aug_white_noise(train_data, 1)
    train_data=np.vstack((train_data,noise_train_data))
    train_labels=np.hstack((train_labels, train_labels))
 
    train_labels=train_labels.reshape(train_labels.shape+(1,))
    train_data=np.hstack((train_data, train_labels))
    train_data=shuffle(train_data)
  
    train_data=train_data[:1290,:-1]
    train_labels=train_labels[:1290,-1]
     
    
test_data=data[n_train_data+n_val_data:n_train_data+n_val_data+n_test_data]
test_labels=labels[n_train_data+n_val_data:n_train_data+n_val_data+n_test_data]
test_labels=test_labels.reshape((test_labels.shape)+(1,))

test_data=np.hstack((test_data,test_labels))

import pickle

with open('test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

#shuffle data
#split train validation test 0.6 0.2 0.2
trainX=train_data=train_data.reshape(train_data.shape+(1,))
trainY=train_labels=to_categorical(train_labels)

valX=val_data=val_data.reshape(val_data.shape+(1,))
valY=val_labels=to_categorical(val_labels)
#Normalise input data?
"""One-hot encode labels """

"""Pass the train data to the model"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if args['vis']==True:
    model=load_model(args['weights'])
    preds=model.predict(trainX)
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca=PCA(n_components=2)
    preds=pca.fit_transform(preds)
    
    print(preds)
    x_flag=0
    s_flag=0
    o_flag=0
    s_size=40
    for i in range(len(preds)):
        # print(train_labels[i])
        if(train_labels[i].argmax()==0): #negative
            color='r'
            if(x_flag==0):
                plt.scatter(preds[i,0], preds[i,1], marker='x', s=s_size, 
                                c=color,label='negative')
                x_flag=1
            plt.scatter(preds[i,0], preds[i,1], marker='x', s=s_size, c=color)
            
        if(train_labels[i].argmax()==1): #neutral
            color='b'
            if(s_flag==0):
                plt.scatter(preds[i,0], preds[i,1], marker='s', s=s_size, 
                                facecolors='none', edgecolors=color,label='neutral')
                s_flag=1
            plt.scatter(preds[i,0], preds[i,1], marker='s', facecolors='none', s=s_size, edgecolors=color)
        elif(train_labels[i].argmax()==2): #positive
            color='g'
            if(o_flag==0):
                plt.scatter(preds[i,0], preds[i,1], marker='o', 
                            s=s_size, facecolors='none', edgecolors=color,label='positive')
                o_flag=1
            plt.scatter(preds[i,0], preds[i,1], marker='o',facecolors='none', s=s_size, edgecolors=color)

    
    plt.legend(prop={'size': 10})
    plt.show()
    exit()
model=Sequential()

model.add(LSTM(256, input_shape=(trainX.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))
model.add(Activation('softmax'))
model.compile(metrics=['acc'], loss='categorical_crossentropy', optimizer='adam')

"""Model Checkpoint"""
model_path='./model'

if(os.path.exists(model_path)):
    for file in os.listdir(model_path):
        os.remove(os.path.join(model_path,file))

if not os.path.exists(model_path):
    os.makedirs(model_path)

model_path=os.path.join(model_path,'model-weights-{epoch:02d}-{loss:.4f}.hdf5')
model_checkpoint_callback=ModelCheckpoint(model_path, monitor='loss', save_best_only=True, mode='min')
model.fit(trainX, trainY, validation_data = (valX, valY), epochs=50, 
                        batch_size=64, callbacks=[model_checkpoint_callback])
                        
