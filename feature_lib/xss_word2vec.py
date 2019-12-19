#Published:2017/07/21
#Ref:https://github.com/SparkSharly/DL_for_xss
import os
import pandas as pd
import csv,pickle,time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from lib.utils import GeneSeg
from gensim.models.word2vec import Word2Vec
import random,json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import tensorflow as tf
class demo:
    def __init__(self):
        """ xss_word2vec module init
        """
        print("[+] Init module xss_word2vec")
        self.vocabulary_size=3000
        self.batch_size=128
        self.embedding_size=128
        self.num_skips=4
        self.skip_window=5
        self.num_sampled=64
        self.num_iter=5
        self.plot_only=100
        self.log_dir="word2vec.log"
        self.plt_dir="file\\word2vec.png"
        self.vec_dir="file\\word2vec.pickle"
        self.pre_datas_train="file\\pre_datas_train.csv"
        self.datas_train="file\\datas_train.csv"
        self.dictionary=None
        self.embeddings=None
        self.reverse_dictionary=None
        self.words=[]
        self.datas=[]

    def pre_processing(self,all_samples_filename):
        print("[+] Start Preprocessing")
        datadir='data/xss'
        filename=os.path.join(datadir,all_samples_filename)
        all_samples=pd.read_csv(filename,header=0,names=['payload','label'])
        print("[+] Load Data")
        return all_samples['payload'],all_samples['label']

    def fxy_train(self,all_X_samples=None,all_Y_samples=None):
        """get features x and y
        Parameters
        ----------
        self.true_X_samples:DataFrame
        self.false_X_samples:DataFrame

        Returns
        ----------
        X:numpy.ndarray
            
        Y:numpy.ndarray
            columns:classes num
        """
        print("[+] Start feature engineering for trainning")
        start=time.time()
        false_X_samples=all_X_samples[all_Y_samples==1].reset_index(drop=True) #multiclass
        true_X_samples=all_X_samples[all_Y_samples==0].reset_index(drop=True)
        for i in range(len(false_X_samples)):
            payload=str(false_X_samples.loc[i])
            word=GeneSeg(payload)
            self.datas.append(word)
            self.words+=word
        #print(self.datas,self.words)
        data_set=self.build_dataset(self.datas,self.words)
        model=Word2Vec(data_set,size=self.embedding_size,window=self.skip_window,negative=self.num_sampled,iter=self.num_iter)
        self.embeddings=model.wv
        self.save(self.embeddings)
        X,Y=self.pre_process(false_X_samples,true_X_samples)
        X=self.generate_vec(X)
        end=time.time()
        print("[+] Done!Successfully got Feature X Y"+" Spend time:",end-start)
        print("[+] Fxy shape:",X.shape,Y.shape)
        return X,Y

    def fxy_test(self,all_X_samples=None,all_Y_samples=None):
        self.fxy_train(all_X_samples,all_Y_samples)
 
    def generate_vec(self,words):
        """ get words embedding vector from words dictionary
        Parameters
        ----------
        words:numpy.ndarray
            word index sequences
        
        Returns
        ----------
        all_data:numpy.ndarray
            words embeddings vector,dimensions=3
        """
        print("[+] Getting word embeddings vector")
        start=time.time()
        all_data=[]
        for word in words:
            data_embed=[]
            for d in word:
                if d != -1:
                    data_embed.append(self.embeddings[self.reverse_dictionary[d]])
                else:
                    data_embed.append([0.0] * len(self.embeddings["UNK"]))
            all_data.append(data_embed)
        all_data=np.array(all_data)
        end=time.time()
        print("[+] Got word embeddings vector"+" Spend time:",end-start)
        # print("[+] Write trian datas to:",self.datas_train)
        # with open(self.datas_train,"w") as f:
        #     for i in range(len(all_data)):
        #         data_line=str(all_data[i])+"|"+str(labels[i].tolist())+"\n"
        #         f.write(data_line)
        
        # end=time.time()
        # print("[+] Write datas over!"+" Spend time: ",end-start)
        return all_data

    def pre_process(self,false_X_samples,true_X_samples):
        """ get words index vector from words dictionary
        Parameters
        ----------
        false_X_samples:DataFrame
            attack payloads
        true_X_samples:DataFrame
            normal datas
        word2vec["dictionary"]:dict
            key=words,value=index of each word in word dictionary
        word2vec["reverse_dictionary"]:dict
            key=index of each word in word dictionary,value=words
        word2vec["embeddings"]:KeyedVectors
            word embedding vectors:embeddings['UNK']
            word index vectors:embeddings.vocab['UNK'].index 
            word nums vectors:embeddings.vocab['UNK'].count #https://codeday.me/bug/20190207/611442.html

        Returns
        ----------
        word2vec["train_size"]:int
            rows of datas
        word2vec["input_num"]:int
            max len of all words sequences
        word2vec["dims_num"]:int
            embedding dimensions=self.embedding_size=128
        datas_index:numpy.ndarray
            words index vector,dimensions=2
        labels:numpy.ndarray
            all data labels
        """
        print("[+] Getting word index vector")
        with open(self.vec_dir,"rb") as f :
            word2vec=pickle.load(f)
            self.dictionary=word2vec["dictionary"]
            self.reverse_dictionary=word2vec["reverse_dictionary"]
            self.embeddings=word2vec["embeddings"]
        xssed_data=[]
        normal_data=[]

        for i in range(len(false_X_samples)):
            payload=str(false_X_samples.loc[i])
            word=GeneSeg(payload)
            xssed_data.append(word)
        
        for i in range(len(true_X_samples)):
            payload=str(true_X_samples.loc[i])
            word=GeneSeg(payload)
            normal_data.append(word)

        xssed_num=len(xssed_data)
        normal_num=len(normal_data)
        xssed_labels=[1]*xssed_num
        normal_labels=[0]*normal_num
        datas=xssed_data+normal_data
        labels=xssed_labels+normal_labels
        labels=to_categorical(labels)
    
        datas_index=[self.to_index(data) for data in datas]
        datas_index=pad_sequences(datas_index,value=-1)
        train_size=len(labels)
      
        input_num=len(datas_index[0])
        dims_num = self.embeddings["UNK"].shape[0]

        word2vec["train_size"]=train_size
        word2vec["input_num"]=input_num # max len in all sequences 
        word2vec["dims_num"]=dims_num # embeddings vector 128
        with open(self.vec_dir,"wb") as f :
            pickle.dump(word2vec,f)
        print("[+] Saved word2vec to:",self.vec_dir)
        # print("Write trian datas to:",self.pre_datas_train)
        # with open(self.pre_datas_train,"w") as f:
        #     for i in range(train_size):
        #         data_line=str(datas_index[i].tolist())+"|"+str(labels[i].tolist())+"\n"
        #         f.write(data_line)
        # print("Write datas over!")
        return datas_index,labels

    def to_index(self,data):
        """ index of words using self.dictionary
        Parameters
        ----------
        data:list
            words of each data,values=words,dimensions=1

        Returns
        ----------
        d_index:list
            index of each word,values=index,dimensions=1
        """
        d_index=[]
        for word in data:
            if word in self.dictionary.keys():
                d_index.append(self.dictionary[word])
            else:
                d_index.append(self.dictionary["UNK"])
        return d_index

    def build_dataset(self,datas,words):
        """ words generalization using Top self.vocabulay_size
        Parameters
        ----------
        datas:list
            all samples datas,dimensions=2
        words:list
            all samples datas words,dimensions=1

        Returns
        ----------
        data_set:list
            generalized datas
        """
        count=[["UNK",-1]]
        counter=Counter(words)
        count.extend(counter.most_common(self.vocabulary_size-1))
        vocabulary=[c[0] for c in count]
        data_set=[]
        for data in datas:
            d_set=[]
            for word in data:
                if word in vocabulary:
                    d_set.append(word)
                else:
                    d_set.append("UNK")
                    count[0][1]+=1
            data_set.append(d_set)
        return data_set

    def save(self,embeddings):
        dictionary=dict([(embeddings.index2word[i],i)for i in range(len(embeddings.index2word))])
        reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
        word2vec={"dictionary":dictionary,"embeddings":embeddings,"reverse_dictionary":reverse_dictionary}
        with open(self.vec_dir,"wb") as f:
            pickle.dump(word2vec,f)
        print("[+] Saved word2vec to:",self.vec_dir)
    