import os
import sys
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import requests
from urllib.parse import unquote
import re
import nltk
from gensim.models.word2vec import Word2Vec
from collections import Counter
from collections import OrderedDict
from collections import defaultdict

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

class tfidf():
    def __init__(self,level="char",ngram=(1,3),decode_error='ignore'):
        self.level=level
        self.ngram=ngram
        self.decode_error=decode_error

        self.dictionary=None
        self.vectorizer=None
        
    def fit_transform(self,train_x='',train_y=''):
        vectorizer = TfidfVectorizer(min_df = 0.0,analyzer=self.level,sublinear_tf=True,decode_error=self.decode_error,ngram_range=self.ngram) 
        fxy_train_x=vectorizer.fit_transform(train_x.values.astype('U')) # judge np.nan
        fxy_train_y=train_y.values
        self.vectorizer=vectorizer
        self.dictionary=vectorizer.vocabulary_
        return fxy_train_x,fxy_train_y

    def transform(self,test_x=''):
        fxy_test_x=self.vectorizer.transform(test_x.values.astype('U'))
        return fxy_test_x

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

def tokenizer(payload):
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
    |'
    |"
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

def text_to_word_sequence(text,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True, split=" ",punctuation=False):
    if lower:
        text = text.lower()
    if not punctuation:

        if sys.version_info < (3,):
            if isinstance(text, unicode):
                translate_map = dict((ord(c), unicode(split)) for c in filters)
                text = text.translate(translate_map)
            elif len(split) == 1:
                translate_map = maketrans(filters, split * len(filters))
                text = text.translate(translate_map)
            else:
                for c in filters:
                    text = text.replace(c, split)
        else:
            translate_dict = dict((c, split) for c in filters)
            translate_map = maketrans(translate_dict)
            text = text.translate(translate_map)

        seq = text.split(split)
    else:
        seq=tokenizer(text)
    return [i for i in seq if i]

class wordindex():
    def __init__(self, num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 char_level=False,
                 oov_token=None,
                 max_length=None,
                 punctuation=False):

        self.word_counts = OrderedDict()
        self.word_docs = defaultdict(int)
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = defaultdict(int)
        self.word_index = dict()
        self.index_word = dict()

        self.max_length=max_length
        self.punctuation=punctuation

        self.input_dim=None

    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split,
                                            self.punctuation)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

    def fit_on_texts(self, texts):
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split,
                                            self.punctuation)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.index_word = dict((c, w) for w, c in self.word_index.items())

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def fit_transform(self,train_x='',train_y=''):
        self.fit_on_texts(train_x)
        train_x = self.texts_to_sequences(train_x)

        if self.max_length:
            fxy_train_x=pad_sequences(train_x,maxlen=self.max_length)
        else:
            fxy_train_x=pad_sequences(train_x)
            self.max_length=len(fxy_train_x[0])

        self.input_dim = len(self.word_index)+1
        fxy_train_y=train_y.values
        
        return fxy_train_x,fxy_train_y

    def transform(self,test_x=''):
        test_x = self.texts_to_sequences(test_x)
        fxy_test_x=pad_sequences(test_x,maxlen=self.max_length)

        return fxy_test_x

class word2vec():
    def __init__(self,pretrain=True,one_class=True,out_dimension=3,vocabulary_size=None,max_length=None,embedding_size=16,skip_window=5,num_sampled=64,num_iter=5,max_log_length=1024):
        self.one_class=one_class
        self.out_dimension=out_dimension
        self.vocabulary_size=vocabulary_size
        self.embedding_size=embedding_size
        self.input_dim=embedding_size
        self.skip_window=skip_window
        self.num_sampled=num_sampled
        self.num_iter=num_iter
        self.max_length=max_length

        self.dictionary=None
        self.reverse_dictionary=None
        self.embeddings=None
        self.dictionary_count=None

        self.embeddings_matrix=None

        self.pretrain=pretrain

    def fit_transform(self,train_x='',train_y=''):
        if self.one_class:
            model_X_samples=train_x[train_y==1].reset_index(drop=True) 
        else:
            model_X_samples=train_x

        # malicious samples tokenizer
        datas=[]
        words=[]
        for i in range(len(model_X_samples)):
            payload=str(model_X_samples.loc[i])
            word=tokenizer(payload)
            datas.append(word)
            words+=word
        
        # Generalization by malicious samples top 3000 word dictionary 
        count=[["UNK",0]]
        counter=Counter(words)
        if self.vocabulary_size is not None:
            count.extend(counter.most_common(self.vocabulary_size-1))
        else:
            count.extend(counter.most_common())
        self.dictionary_count=count
        
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
        
        # Word2Vec model
        model=Word2Vec(data_set,size=self.embedding_size,window=self.skip_window,negative=self.num_sampled,iter=self.num_iter)
        self.embeddings=model.wv

        # get pretrain maxtrix
        word2idx = {"_PAD": 0}
        vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
        self.embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
        for i in range(len(vocab_list)):
            word = vocab_list[i][0]
            word2idx[word] = i + 1
            self.embeddings_matrix[i + 1] = vocab_list[i][1]

        # word2seq
        train_seq=[]
        for i in range(len(train_x)):
            payload=str(train_x.loc[i])
            word=tokenizer(payload)
            train_seq.append(word)

        self.dictionary=dict([(self.embeddings.index2word[i],i) for i in range(len(self.embeddings.index2word))])
        self.reverse_dictionary={v:k for k,v in self.dictionary.items()}

        #word2index
        train_index=self._index(train_seq)
        if self.max_length:
            train_index=pad_sequences(train_index,maxlen=self.max_length,value=0)
        else:
            train_index=pad_sequences(train_index,value=0)
            self.max_length=len(train_index[0])
        # if pretrain or not    
        if self.pretrain:
            self.input_dim=self.embeddings_matrix.shape[0]
            return train_index,train_y.values

        #word2vec
        if self.out_dimension==3:
            fxy_train_x=self._vec3(train_index)
        else:
            fxy_train_x=self._vec2(train_index)

        fxy_train_y=train_y.values
        return fxy_train_x,fxy_train_y

    def transform(self,test_x=''):
        test_seq=[]
        # tokenizer
        for i in range(len(test_x)):
            payload=str(test_x.loc[i])
            word=tokenizer(payload)
            test_seq.append(word)
        # index
        test_index=self._index(test_seq)
        test_index=pad_sequences(test_index,maxlen=self.max_length,value=0)
        if self.pretrain:
            return test_index
        # vec
        if self.out_dimension==3:
            fxy_test_x=self._vec3(test_index)
        else:
            fxy_test_x=self._vec2(test_index)

        return fxy_test_x

    def _index(self,seq):
        all_index=[]
        for x in seq:
            index=[]
            for word in x:
                if word in self.dictionary.keys():
                    index.append(self.dictionary[word])
                else:
                    index.append(0)
            all_index.append(index)
        return all_index

    def _vec3(self,index):
        all_vec=np.zeros(shape=(len(index),len(index[0]),self.embedding_size))
        j=0
        for x in index:
            vec=np.zeros(shape=(len(index[0]),self.embedding_size))
            i=0
            for word in x:
                if word!=0:
                    vec[i]=self.embeddings[self.reverse_dictionary[word]]
                    #vec.append(self.embeddings[self.reverse_dictionary[word]])
                else:
                    vec[i]=[0.0]*self.embedding_size
                    #vec.append([0.0]*len(self.embeddings['UNK']))
                i=i+1
            all_vec[j]=vec
            j=j+1
            #all_vec.append(vec)
        return all_vec

    def _vec2(self,index):
        all_vec=np.zeros(shape=(len(index),len(index[0])*self.embedding_size))
        j=0
        for x in index:
            vec=np.zeros(shape=len(index[0])*self.embedding_size)
            i=0
            for word in x:
                if word!=0:
                    vec[i:i+self.embedding_size]=self.embeddings[self.reverse_dictionary[word]]
                    #vec.extend(self.embeddings[self.reverse_dictionary[word]])
                else:
                    vec[i:i+self.embedding_size]=[0.0]*self.embedding_size
                    #vec.extend([0.0]*len(self.embeddings['UNK']))
                i=i+self.embedding_size
            #all_vec.append(vec)
            all_vec[j]=vec
            j=j+1
        return all_vec

