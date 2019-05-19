import pandas as pd
# import nltk
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
import csv
data = pd.read_csv('Train.csv')
x = data.values[:, 0]
y = data.values[:, 1]
c = 0
for t in y:
    if t == 'pos':
        y[c] = 1
    else:
        y[c] = 0
    c += 1
y = to_categorical(y)
train_data = pd.read_csv('Test.csv')
x_test = train_data.values[:, 0]

x = x.reshape((-1, 1))
for t in range(x.shape[0]):
    x[t] = ' '.join(x[t][0].split()[:20])

x = x.reshape((-1,))

x_test = x_test.reshape((-1, 1))
for t in range(x_test.shape[0]):
    x_test[t] = ' '.join(x_test[t][0].split()[:20])

x_test = x_test.reshape((-1,))


def create_dictoionaty_vec():
    f = open('glove.6B.50d.txt', encoding='utf-8')
    embedding_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float')
        embedding_index[word] = vector
    f.close()
    return embedding_index


vector = create_dictoionaty_vec()


def embedding_output(x):
    embedding_out = np.zeros((x.shape[0], 20, 50))
    for ix in range(x.shape[0]):
        x[ix] = x[ix].split()
        for ij in range(20):
            try:
                embedding_out[ix][ij] = vector[x[ix][ij].lower()]
            except:
                embedding_out[ix][ij] = np.zeros((50,))
    return embedding_out


train_data = embedding_output(x)
test = embedding_output(x_test)


def create_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 50), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


def train():
    checkpoint = ModelCheckpoint(
        'best_model.h5', monitor='val_acc', save_best_only=True, verbose=True)
    earlystoping = EarlyStopping(patience=10, monitor='val_acc')
    hist = model.fit(train_data, y, epochs=100, batch_size=64, shuffle=True,
                     validation_split=0.2, callbacks=[checkpoint, earlystoping])


model = create_model()
model.load_weights('best_model.h5')
pred = model.predict_classes(test)

with open('submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Id', 'label'])
    for i in range(10000):
        if pred[i] == 0:
            t = 'pos'
        else:
            t = 'neg'
        w.writerow([i, t])
