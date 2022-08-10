import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df_eeg_data=pd.read_csv('emotions.csv')

data=df_eeg_data.iloc[:,:-1]
labels=df_eeg_data.iloc[:,-1]
data=data.values
labels=labels.values

print(np.max(data), np.min(data))

interval=5
cnt=1
emotion='NEUTRAL'
if not os.path.exists(emotion.lower()):
    os.mkdir(emotion.lower())
for i in range(len(data)):
    if(labels[i]==emotion):
        target=data[i]    
        bins=[]
        num=-100
        while(num<=100):
            bins.append(num)
            num+=interval
        plt.figure()
        # ax = fig.add_subplot(111)
        plt.hist(target, bins=bins)
        title=f'eeg_{i+1}'
        plt.title(title)
        plt.savefig('./'+emotion.lower()+'/'+title+'.png')
        cnt+=1
    if(cnt>9): break
# plt.show()
