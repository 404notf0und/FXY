import numpy as np
from feature_vec.data import data_load
from feature_vec.nlp2vec import tfidf,wordindex,word2vec
from sklearn.model_selection import train_test_split
from feature_vec.model import lstm,textcnn
from sklearn.metrics import classification_report

# data load
x1,y1,x2,y2=data_load('part1.csv','part3.csv')

# feature engineering
nlp=word2vec(one_class=False,pretrain=True) # init feature class
#nlp=wordindex(punctuation=True)
fx1,fy1=nlp.fit_transform(x1,y1) # training data to vec
fx2=nlp.transform(x2) # test data to vec
weights=nlp.embeddings_matrix

# model training
train_x, valid_x, train_y, valid_y = train_test_split( fx1, fy1, random_state=2019,test_size = 0.2) 
model=lstm(input_type='word2vec_pretrain',max_len=nlp.max_length,input_dim=nlp.input_dim,output_dim=16,weight_matrix=weights)
model.fit(train_x, train_y, validation_data=(valid_x,valid_y), epochs=1, batch_size=128)

# model testing
r2=np.asarray(model.predict(fx2)).round()

# model evaluation
print(classification_report(y2,r2))
