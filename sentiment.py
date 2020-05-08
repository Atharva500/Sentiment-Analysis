import numpy as np
import pandas as pd
import os
import pandas
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,Dense,Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sklearn
from sklearn.model_selection import train_test_split


reader = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='latin-1')
y = reader.iloc[:,0]
X = reader.iloc[:,-1]
y = np.array(y)
X = np.array(X)

y = y.reshape((1599999,1))
X = X.reshape((1599999,1))

X = X.tolist()
y = y.tolist()

temp = X
X = [' '.join(x) for x in temp]

for i in range(len(y)):
    y[i]=y[i][0]/2

tokenizer = Tokenizer(num_words=200,oov_token="<OOV>")
tokenizer.fit_on_texts(X)


seq = tokenizer.texts_to_sequences(X)
pad = pad_sequences(seq,padding='post')

print(pad.shape)

x_train,x_test,y_train,y_test = train_test_split(pad,y,test_size=0.5)

from keras.layers import LSTM
model = Sequential([
    Embedding(4000,1024,input_length=118),
    Bidirectional(LSTM(256,return_sequences=True)),
    Flatten(),
    Dense(units=3,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=1024,verbose=1)
model.save('/kaggle/working/saved_model/model.h5')
