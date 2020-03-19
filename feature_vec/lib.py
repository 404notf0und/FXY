from keras.preprocessing.sequence import pad_sequences

def train_test_max_align(train_x,test_x,maxlen=None):
    train_len=len(train_x[0])
    test_len=len(test_x[0])
    max_len=max(train_len,test_len)
    if train_len<test_len:
        train_x=pad_sequences(train_x, maxlen=test_len)
    elif train_len>test_len:
        test_x=pad_sequences(test_x, maxlen=train_len)
    return train_x,test_x
