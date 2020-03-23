from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Dropout,Input,InputLayer,SpatialDropout1D,Conv1D,MaxPool1D,Flatten,concatenate
from keras.layers.embeddings import Embedding

def lstm_2D(max_len,input_dim,output_dim):
    model = Sequential()
    model.add(Embedding(input_dim, output_dim, input_length=max_len))
    model.add(Dropout(0.5))
    model.add(LSTM(64, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def lstm_3D(max_len,input_dim):
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len,input_dim)))
    model.add(Dropout(0.5))
    model.add(LSTM(64, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# def _CNN(max_log_length,input_dim,output_dim):
#     # word level
#     kernel_size = [2,3,4,5]
#     acti = 'relu'
#     my_input = Input(shape=(max_log_length,), dtype='int32')
#     emb = Embedding(input_dim, output_dim, input_length=max_log_length)(my_input)
#     emb = SpatialDropout1D(0.2)(emb)

#     net = []
#     for kernel in kernel_size:
#         con = Conv1D(32, kernel, activation=acti, padding="same")(emb)
#         con = MaxPool1D(2)(con)
#         net.append(con)
#     net = concatenate(net, axis=-1)
#     net = Flatten()(net)
#     net = Dropout(0.5)(net)
#     net = Dense(64, activation='relu')(net)
#     net = Dropout(0.5)(net)
#     net = Dense(1, activation='softmax')(net)
#     model = Model(inputs=my_input, outputs=net)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

