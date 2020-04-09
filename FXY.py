from feature_vec.lib import data_load,model_metrics
from feature_vec.nlp2vec import tfidf,wordindex,word2vec
from sklearn.model_selection import train_test_split
from feature_vec.model import lstm,textcnn

# data load
x1,y1=data_load('part6A_spamail_A.csv')
x2,y2=data_load('part6A_spamail_B.csv')

# feature engineering
#nlp=word2vec(one_class=False,tunning=True,punctuation='concise') # init feature class
nlp=wordindex(char_level=False)
fx1,fy1=nlp.fit_transform(x1,y1) # training data to vec
fx2=nlp.transform(x2) # test data to vec
#weight_matrix=nlp.embeddings_matrix

# model training
train_x, valid_x, train_y, valid_y = train_test_split( fx1, fy1, random_state=2019,test_size = 0.3) 
#model=textcnn(input_type='word2vec_tunning',max_len=nlp.max_length,input_dim=nlp.input_dim,output_dim=16,class_num=1,weight_matrix=weight_matrix)
#model=textcnn(input_type='word2vec',max_len=nlp.max_length,input_dim=nlp.input_dim,output_dim=16,class_num=1)
model=textcnn(input_type='wordindex',max_len=nlp.max_length,input_dim=nlp.input_dim,output_dim=16,class_num=1)
model.fit(train_x, train_y, validation_data=(valid_x,valid_y), epochs=10, batch_size=128)

# model testing
pre=model.predict(fx2)

# model evaluation
model_metrics(y2,pre)
