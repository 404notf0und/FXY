import os
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from .data import data_load
import requests
from urllib.parse import unquote
import re
import nltk
from collections import Counter
from gensim.models.word2vec import Word2Vec
from .lib import train_test_max_align
class tfidf():
    def __init__(self,level="char",ngram=(1,3),decode_error='ignore'):
        self.level=level
        self.ngram=ngram
        self.decode_error=decode_error

        self.vocabulary=None
        self.vectorizer=None

        self.train=False
        self.test=False
        
        self.fxy_train_x=None
        self.fxy_train_y=None
        self.fxy_test_x=None
        self.fxy_test_y=None
        
    def fit_vec(self,train_x='',train_y='',test_x='',test_y=''):
        if len(train_x)!=0:
            self.train=True
        if len(test_x)!=0:
            self.test=True
        if self.train:
            vectorizer = TfidfVectorizer(min_df = 0.0,analyzer=self.level,sublinear_tf=True,decode_error=self.decode_error,ngram_range=self.ngram) 
            self.fxy_train_x=vectorizer.fit_transform(train_x.values.astype('U')) # judge np.nan
            self.fxy_train_y=train_y.values
        if self.test:
            self.fxy_test_x=vectorizer.transform(test_x.values.astype('U'))
            self.fxy_test_y=test_y.values
        print("[+] Fxy shape:",self.fxy_train_x,self.fxy_train_y,self.fxy_test_x.shape,self.fxy_test_y.shape)
        return self.fxy_train_x,self.fxy_train_y,self.fxy_test_x,self.fxy_test_y

    # def plot(self,vec_x=None,vec_y=None):
    #     svd = TruncatedSVD(n_components=2,random_state=2020)
    #     data_svd=svd.fit_transform(vec_x)
    #     #data_svd_label=pd.concat([pd.DataFrame(data_svd),self.orgin_y],ignore_index=True,axis=1)
    #     fig, ax = plt.subplots()
    #     type1=None
    #     type2=None
    #     for i in range(int(data_svd.size/2)):
    #         x = data_svd[i]
    #         if vec_y[i]==0:
    #             type1=ax.scatter(x[0], x[1],c='red')
    #             #ax.annotate(self.orgin_x[i], (x[0], x[1]))
    #         else:
    #             type2=ax.scatter(x[0], x[1],c='blue')
    #             #ax.annotate(self.orgin_x[i], (x[0], x[1]))
        
    #     plt.legend((type1, type2),('normal','malicious'))
    #     plt.show()


class wordindex():
    def __init__(self,level='char',max_log_length=None,input_dim=None):
        self.level=level
        self.max_log_length=max_log_length
        self.input_dim=input_dim
        self.tokenizer=None

        self.train=False
        self.test=False

        self.fxy_train_x=None
        self.fxy_train_y=None
        self.fxy_test_x=None
        self.fxy_test_y=None

    def fit_vec(self,train_x='',train_y='',test_x='',test_y=''):
        if len(train_x)!=0:
            self.train=True
        if len(test_x)!=0:
            self.test=True
        if self.level=='char':
            char_level=True
        else:
            char_level=False
        train_max_log_length=0
        test_max_log_length=0
        if self.train:
            tokenizer = Tokenizer(filters='\t\n', char_level=char_level)
            tokenizer.fit_on_texts(train_x)
            self.input_dim = len(tokenizer.word_index)+1
            train_x = tokenizer.texts_to_sequences(train_x)
            train_index=pad_sequences(train_x)
            self.fxy_train_y=(train_y.values)
            
        if self.test:
            tokenizer.fit_on_texts(test_x)
            self.input_dim = len(tokenizer.word_index)+1
            test_x = tokenizer.texts_to_sequences(test_x)
            test_x=pad_sequences(test_x)
            self.fxy_train_x,self.fxy_test_x=train_test_max_align(train_x,test_x)
            self.fxy_test_y=(test_y.values)


        self.tokenizer=tokenizer
        self.fxy_train_x=np.array(self.fxy_train_x)
        self.fxy_train_y=np.array(self.fxy_train_y)
        self.fxy_test_x=np.array(self.fxy_test_x)
        self.fxy_test_y=np.array(self.fxy_test_y)
        print("[+] Fxy shape:",self.fxy_train_x.shape,self.fxy_train_y.shape,self.fxy_test_x.shape,self.fxy_test_y.shape)
        return self.fxy_train_x,self.fxy_train_y,self.fxy_test_x,self.fxy_test_y

class word2vec():
    def __init__(self,out_dimension=3,vocabulary_size=300,embedding_size=128,skip_window=5,num_sampled=64,num_iter=5,max_log_length=1024):
        self.out_dimension=out_dimension
        self.vocabulary_size=vocabulary_size
        self.embedding_size=embedding_size
        self.skip_window=skip_window
        self.num_sampled=num_sampled
        self.num_iter=num_iter
        self.input_dim=embedding_size
        
        self.words=[]
        self.datas=[]

        self.train=False
        self.test=False

        self.fxy_train_x=None
        self.fxy_train_y=None
        self.fxy_test_x=None
        self.fxy_test_y=None

        self.dictionary=None
        self.reverse_dictionary=None
        self.embeddings=None

    def tokenizer(self,payload):
        #数字泛化为"0"
        payload=payload.lower()
        payload=unquote(unquote(payload))
        payload,num=re.subn(r'\d+',"0",payload)
        #替换url为”http://u
        payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
        #分词
        #r = r'\w+'
        r = '''
        (?x)[\w\.]+?\(
        |\)
        |\'
        |\"
        |\@
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
        '''
        return nltk.regexp_tokenize(payload, r)

    def fit_vec(self,train_x='',train_y='',test_x='',test_y=''):
        if len(train_x)!=0:
            self.train=True
        if len(test_x)!=0:
            self.test=True
        train_max_log_length=0
        test_max_log_length=0
        if self.train:
            false_X_samples=train_x[train_y==1].reset_index(drop=True) #multiclass
            true_X_samples=train_x[train_y==0].reset_index(drop=True)
            # malicious samples tokenizer
            for i in range(len(false_X_samples)):
                payload=str(false_X_samples.loc[i])
                word=self.tokenizer(payload)
                self.datas.append(word)
                self.words+=word
            # Generalization by malicious samples top 3000 word dictionary 
            count=[["UNK",-1]]
            counter=Counter(self.words)
            count.extend(counter.most_common(self.vocabulary_size-1))
            vocabulary=[c[0] for c in count]
            data_set=[]
            for data in self.datas:
                d_set=[]
                for word in data:
                    if word in vocabulary:
                        d_set.append(word)
                    else:
                        d_set.append("UNK")
                        count[0][1]+=1
                data_set.append(d_set)
            # Word2Vec
            model=Word2Vec(data_set,size=self.embedding_size,window=self.skip_window,negative=self.num_sampled,iter=self.num_iter)
            self.embeddings=model.wv
            # word2seq
            train_seq=[]
            for i in range(len(train_x)):
                payload=str(train_x.loc[i])
                word=self.tokenizer(payload)
                train_seq.append(word)
            self.dictionary=dict([(self.embeddings.index2word[i],i) for i in range(len(self.embeddings.index2word))])
            self.reverse_dictionary={v:k for k,v in self.dictionary.items()}
            #word2index
            train_index=self._index(train_seq)
            
        if self.test:
            test_seq=[]
            for i in range(len(test_x)):
                payload=str(test_x.loc[i])
                word=self.tokenizer(payload)
                test_seq.append(word)
            test_index=self._index(test_seq)
            train_index,test_index=train_test_max_align(train_index,test_index)

        if self.train:
            self.input_dim=self.embeddings["UNK"].shape[0]
            #word2Vectorization
            if self.out_dimension==3:
                train_vec=self._vec3(train_index)
            else:
                train_vec=self._vec2(train_index)
            self.fxy_train_x=np.array(train_vec)
            self.fxy_train_y=np.array(train_y)
            print("[+] Fxy shape:",self.fxy_train_x.shape,self.fxy_train_y.shape)

        if self.test:
            if self.out_dimension==3:
                test_vec=self._vec3(test_index)
            else:
                test_vec=self._vec2(test_index)
            self.fxy_test_x=np.array(test_vec)
            self.fxy_test_y=np.array(test_y)
        print("[+] Fxy shape:",self.fxy_train_x,self.fxy_train_y,self.fxy_test_x.shape,self.fxy_test_y.shape)
        return self.fxy_train_x,self.fxy_train_y,self.fxy_test_x,self.fxy_test_y
    def _index(self,seq):
        all_index=[]
        for x in seq:
            index=[]
            for word in x:
                if word in self.dictionary.keys():
                    index.append(self.dictionary[word])
                else:
                    index.append(self.dictionary["UNK"])
            all_index.append(index)
        all_index=pad_sequences(all_index)
        return all_index
    def _vec3(self,index):
        all_vec=[]
        for x in index:
            vec=[]
            for word in x:
                if word!=-1:
                    vec.append(self.embeddings[self.reverse_dictionary[word]])
                else:
                    vec.append([0.0]*len(self.embeddings['UNK']))
            all_vec.append(vec)
        return all_vec
    def _vec2(self,index):
        all_vec=[]
        for x in index:
            vec=[]
            for word in x:
                if word!=-1:
                    vec.extend(self.embeddings[self.reverse_dictionary[word]])
                else:
                    vec.extend([0.0]*len(self.embeddings['UNK']))
            all_vec.append(vec)
        return all_vec

# x1,y1,x2,y2=data_load('url.csv','url3.csv')
# tf=tfidf()
# fx1,fy1,fx2,fy2=tf.feature_vec(train_x=x1,train_y=y1,test_x=x2,test_y=y2)