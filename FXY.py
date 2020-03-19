from feature_vec.data import data_load
from feature_vec.nlp_vec import tfidf,wordindex,word2vec
from feature_vec.model import _LSTM,_CNN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

x1,y1,x2,y2=data_load('part1.csv','part2.csv')
wd=wordindex(level='word')
fx1,fy1,fx2,fy2=wd.fit_vec(x1,y1,x2,y2)
train_x, valid_x, train_y, valid_y = train_test_split( fx1, fy1, random_state=2019,test_size = 0.2) 
#model=_CNN(wd.max_log_length,wd.input_dim,wd.output_dim)
#model.fit(train_x, train_y, validation_data=(valid_x,valid_y), epochs=1, batch_size=128)
lr=LogisticRegression()
lr.fit(train_x,train_y)
r1=lr.predict(fx2)
print(classification_report(fy2,r1))