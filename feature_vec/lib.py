import os
import pandas as pd
import requests
from urllib.parse import unquote
import numpy as np
from sklearn.metrics import classification_report

def data_load(file=None):
    datadir='./data/'
    x=''
    y=''
    if file:
        file=os.path.join(datadir,file)
        samples=pd.read_csv(file)
        samples=samples.sample(frac=1)
        x=samples['payload'].astype('str')
        x=x.apply(decode)
        if 'label' in samples.columns:
            y=samples['label']
    return x,y

def decode(payload):
    return unquote(unquote((payload)))

def model_metrics(true,false):
    if false.shape[1]>2:
        #input:pre=(x,class) true=(x,class) or true=(x,)
        #output:pre=(x,) true=(x,)
        pre=[]
        for i in range(false.shape[0]):
            pre.append(np.argmax(false,axis = 1)[i])
        pre=np.array(pre)
        if true.shape!=(true.shape[0],):
            true=np.argmax(true, axis=1)
    else:
        #input:pre=(x,class) true=(x,)
        #output:pre=(x,) true=(x,)
        pre=false.round()
    print(classification_report(true,pre))
    pass


    