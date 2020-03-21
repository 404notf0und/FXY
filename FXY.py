from feature_vec.data import data_load
from feature_vec.nlp2vec import tfidf,wordindex,word2vec
from feature_vec.model import _LSTM,_CNN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

x1,y1,x2,y2=data_load('part1.csv','part3.csv')
nlp=wordindex(level='word')
#nlp=tfidf(level='word')
#nlp=wordindex(level='char')
#nlp=wordindex(level='word')
#nlp=word2vec(level='char')
#nlp=word2vec(level='word')
fx1,fy1=nlp.fit_transform(x1,y1)
fx2=nlp.transform(x2)
#print(nlp.dictionary_count)
lr=LogisticRegression()
lr.fit(fx1,fy1)
r2=lr.predict(fx2)
print(classification_report(y2,r2))
#train_x, valid_x, train_y, valid_y = train_test_split( fx1, fy1, random_state=2019,test_size = 0.2) 
#model=_CNN(wd.max_log_length,wd.input_dim,wd.output_dim)
#model.fit(train_x, train_y, validation_data=(valid_x,valid_y), epochs=1, batch_size=128)