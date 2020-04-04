import os
import pandas as pd
import requests
from urllib.parse import unquote

def data_load(trainfile=None,testfile=None):
    datadir='./data/'
    train_x=''
    train_y=''
    test_x=''
    test_y=''
    if trainfile:
        train_file=os.path.join(datadir,trainfile)
        train_samples=pd.read_csv(train_file)
        #train_samples=train_samples.sample(frac=0.3).reset_index() 
        train_x=train_samples['payload'].astype('str')
        train_x=train_x.apply(decode)
        train_y=train_samples['label']
    if testfile:
        test_file=os.path.join(datadir,testfile)
        test_samples=pd.read_csv(test_file)
        #test_samples=test_samples.sample(frac=0.3).reset_index() 
        test_x=test_samples['payload'].astype('str')
        test_x=test_x.apply(decode)
        if 'label' in test_samples.columns:
            test_y=test_samples['label']
            return train_x,train_y,test_x,test_y 
        else:
            return train_x,train_y,test_x,test_y
    return train_x,train_y,test_x,test_y

def decode(payload):
    return unquote(unquote((payload)))


    