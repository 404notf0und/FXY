#Time:2017/10/04
#Ref:https://www.cdxy.me/?p=775

import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding

class demo:
    def __init__(self):
        print("[+] Init module xss_word2vec")
        # self.false_X_samples=false_X_samples
        # self.false_Y_samples=false_Y_samples
        # self.true_X_samples=true_X_samples
        # self.true_Y_samples=true_Y_samples
        # self.all_X_samples=all_X_samples
        # self.all_Y_samples=all_Y_samples

        self.max_log_length=1024
        self.input_dim=3000
        self.output_dim=32
        self.tokenizer=None

    def pre_processing(self,all_samples_filename):
        datadir='data/url'
        filename=os.path.join(datadir,all_samples_filename)
        all_samples=pd.read_csv(filename,header=0,names=['payload','label'])
        return all_samples['payload'],all_samples['label']

    def fxy_train(self,all_X_samples=None,all_Y_samples=None):
        """ feature engineering for trainning
        Parameters
        ----------
        all_X_samples:DataFrame
        all_Y_samples:DataFrame

        Returns
        ----------
        X:numpy.ndarray
        Y:numpy.ndarray
            column dimensions=1
        """
        print("[+] Start feature engineering for trainning")
        tokenizer = Tokenizer(filters='\t\n', char_level=True)
        tokenizer.fit_on_texts(all_X_samples)
        self.input_dim = len(tokenizer.word_index)+1
        X = tokenizer.texts_to_sequences(all_X_samples)
        X = pad_sequences(X, maxlen=self.max_log_length)
        #Y=to_categorical(self.all_Y_samples.values)
        Y=all_Y_samples.values
        print("[+] Done!Successfully got Feature X Y")
        print("[+] Fxy shape:",X.shape,Y.shape)
        return X,Y

    def fxy_test(self,all_X_samples=None,all_Y_samples=None):
        """ feature engineering for testing
        Parameters
        ----------
        all_X_samples:DataFrame
        all_Y_samples:DataFrame

        Returns
        ----------
        X:numpy.ndarray
        Y:numpy.ndarray
        """
        print("[+] Start feature engineering for testing")
        tokenizer = Tokenizer(filters='\t\n', char_level=True)
        tokenizer.fit_on_texts(all_X_samples)
        self.input_dim = len(tokenizer.word_index)+1
        X = tokenizer.texts_to_sequences(all_X_samples)
        X = pad_sequences(X, maxlen=self.max_log_length)
        #Y=to_categorical(self.all_Y_samples.values)
        Y=all_Y_samples.values
        print("[+] Done!Successfully got Feature X Y")
        print("[+] Fxy shape:",X.shape,Y.shape)
        return X,Y

    def model_train(self,X,Y):
        """ LSTM
        Parameters
        ----------
        input_dim:int
            字典长度
        out_dim:int
            全连接嵌入的维度
        max_log_length:int
            输入序列的长度

        Returns
        ----------
        model:model
            LSTM model
        """
        print("[+] Start training Using LSTM")
        model = Sequential()
        model.add(Embedding(self.input_dim, self.output_dim, input_length=self.max_log_length)) # embeddings
        model.add(Dropout(0.5))
        model.add(LSTM(64, recurrent_dropout=0.5))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, Y, validation_split=0.25, epochs=3, batch_size=128)
        print("[+] Training Done!")
        return model