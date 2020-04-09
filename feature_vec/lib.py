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
        samples=samples.sample(frac=1,random_state=2020).reset_index(drop=True)
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

# data_clean
# webshell samples traversal
def dir_file(dir_path,label):
    results=[]
    labels=[]
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for name in filenames:
            #print(name)
            if '.php' in name:
                path=(dirpath+'/'+name)
                labels.append(label)
                result=[]
                with open(path,encoding='utf8',errors='ignore') as f:
                    file=f.read().replace('\n',' ').split(' ')
                    file=[l.strip() for l in file if l]
                files=' '.join(file)
                results.append(files)
    
    data=pd.DataFrame(results)
    labels=pd.DataFrame(labels)
    result=pd.concat([data,labels],axis=1,ignore_index=True)
    result.columns=['payload','label']
    return result
#white=dir_file('C:\\Users\\Administrator\\Desktop\\typecho-master\\typecho-master',0)
# print(white.shape)
# white2=dir_file('C:\\Users\\Administrator\\Desktop\\WordPress-master\\WordPress-master',0)
# print(white2.shape)
# black=dir_file('C:\\Users\\Administrator\\Desktop\\webshell-sample-master\\webshell-sample-master',1)
# print(black.shape)
# result=pd.concat([white,white2,black],ignore_index=True)
# result.to_csv('part5A_webshell.csv',encoding='utf_8_sig',index=None)

def dir_file_mix(dir_path):
    results=[]
    labels=[]
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for name in filenames:
            #print(name)
            path=(dirpath+'/'+name)
            if 'spm' in name:
                labels.append(1)
            else:
                labels.append(0)
            result=[]
            with open(path,encoding='utf8',errors='ignore') as f:
                file=f.read().replace('\n',' ').split(' ')
                file=[l.strip() for l in file if l]
            files=' '.join(file)
            results.append(files)
    
    data=pd.DataFrame(results)
    labels=pd.DataFrame(labels)
    result=pd.concat([data,labels],axis=1,ignore_index=True)
    result.columns=['payload','label']
    return result
#mix_data=dir_file_mix('D:\\Documents\\GitHub\\LJDM\\ling-spam\\train-mails')      
#mix_data.to_csv('part6A_spamail.csv',encoding='utf_8_sig',index=None)        
#mix_data2=dir_file_mix('D:\\Documents\\GitHub\\LJDM\\ling-spam\\test-mails')      
#mix_data2.to_csv('part6B_spamail.csv',encoding='utf_8_sig',index=None)   