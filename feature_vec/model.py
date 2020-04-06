from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Dropout,Input,InputLayer,SpatialDropout1D,Conv1D,MaxPool1D,Flatten,concatenate
from keras.layers.embeddings import Embedding

def lstm(max_len,input_dim,output_dim=None,weight_matrix=None,input_type='wordindex',class_num=1):
    model = Sequential()
    if input_type=='wordindex':
        model.add(Embedding(input_dim, output_dim, input_length=max_len))
    if input_type=='word2vec':
        model.add(InputLayer(input_shape=(max_len,input_dim)))
    if input_type=='word2vec_pretrain':
        model.add(Embedding(input_dim, output_dim, input_length=max_len, weights=[weight_matrix], trainable=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def textcnn(max_len,input_dim,output_dim=None,weight_matrix=None,input_type='wordindex',class_num=1):
    kernel_size = [2,3,4,5]
    if input_type=='wordindex':
        my_input = Input(shape=(max_len,))
        emb = Embedding(input_dim, output_dim, input_length=max_len)(my_input)
        emb = SpatialDropout1D(0.2)(emb)
    if input_type=='word2vec':
        my_input = Input(shape=(max_len,input_dim))
        emb = SpatialDropout1D(0.2)(my_input)
    if input_type=='word2vec_pretrain':
        my_input = Input(shape=(max_len,))
        emb = Embedding(input_dim, output_dim, input_length=max_len, weights=[weight_matrix], trainable=True)(my_input)
        emb = SpatialDropout1D(0.2)(emb)

    net = []
    for kernel in kernel_size:
        con = Conv1D(32, kernel, activation='relu', padding="same")(emb)
        con = MaxPool1D(2)(con)
        net.append(con)
    net = concatenate(net, axis=-1)
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(64, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(class_num, activation='sigmoid')(net)
    model = Model(inputs=my_input, outputs=net)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

