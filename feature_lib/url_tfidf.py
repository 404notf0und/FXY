#Published:2017/04/05
#Ref:https://github.com/exp-db/AI-Driven-WAF
#tfidf:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import to_categorical
import pandas as pd

class demo:
    def __init__(self,level="char",ngram=(1,3),decode_error='ignore'):
        """ tfidf init
        Parameters
        ----------
        level:str
            char or word
        ngram:tuple
            (1,2) or (1,3) etc
        decode_error:str
            decode error
        vocabulary:Mapping
            word and word index
        vectorizer:class
            tfidf class
        """
        print("[+] Init module url_tfidf")
        self.level=level
        self.ngram=ngram
        self.decode_error=decode_error

        self.vocabulary=None
        self.vectorizer=None

    def pre_processing(self,all_samples_filename):
        datadir='data/url'
        filename=os.path.join(datadir,all_samples_filename)
        all_samples=pd.read_csv(filename,header=0,names=['payload','label'])
        return all_samples['payload'],all_samples['label']
        
    def fxy_train(self,all_X_samples=None,all_Y_samples=None):
        """get features x and y from trainning data
        Parameters
        ----------
        all_X_samples:DataFrame
        all_Y_samples:DataFrame

        Returns
        ----------
        X:scipy.sparse.csr.csr_matrix
            
        Y:numpy.ndarray
            columns:classes num,column dimensions=2
        """
        print("[+] Start feature engineering for trainning")
        vectorizer = TfidfVectorizer(min_df = 0.0,analyzer=self.level,sublinear_tf=True,decode_error=self.decode_error,ngram_range=self.ngram) 
        self.vectorizer=vectorizer
        X=vectorizer.fit_transform(all_X_samples.values.astype('U')) # judge np.nan
        self.vocabulary=vectorizer.vocabulary_
        Y=to_categorical(all_Y_samples.values)
        print("[+] Successfully got Feature X Y for trainning")
        print("[+] Fxy shape:",X.shape,Y.shape)
        return X,Y

    def fxy_test(self,all_X_samples=None,all_Y_samples=None):
        """ get features x and y from testing data
        Parameters
        ----------
        all_X_samples:DataFrame
        all_Y_samples:DataFrame

        Returns
        ----------
        X:scipy.sparse.csr.csr_matrix
            
        Y:numpy.ndarray
            columns:classes num
        """
        print("[+] Start feature engineering for testing")
        X=self.vectorizer.fit_transform(all_X_samples)
        self.vocabulary=self.vectorizer.vocabulary_
        Y=to_categorical(all_Y_samples.values)
        print("[+] Successfully got Feature X Y for testing")
        print("[+] Fxy shape:",X.shape,Y.shape)
        return X,Y

    def model_train(self,X,Y):
        pass

    def model_test(self,X,Y):
        pass

    def plot(self):
        pass
