#%%
#1. Import Packages
import re, os, datetime, json, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from tensorflow.keras.utils import plot_model

#%%
#2. Data Loading
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')

#%%
#3. Data Inspection
df.info()
df.describe()
df.head()

#%%
#check NaNs and duplicates data
df.isna().sum() #no missing values
df.duplicated().sum() #99 duplicated data

#%%
#4. Data Cleaning

x = df['text']
y = df['category']

temp = []
for idx, text in enumerate(x):
    x[idx] = re.sub('[^a-zA-Z]',' ',text).lower()
    temp.append(len(x[idx].split()))

#%%
#remove duplicates data
df1=pd.concat([x,y],axis=1)
df1=df1.drop_duplicates()

#%%
# double check duplicates data
df1.duplicated().sum() #0 duplicates

#%%
#5. Features Selection
x=df1['text']
y=df1['category']

#%%
#6. Data Pre-Processing
#Tokenization (convert text into numbers)
#Features x
num_words = 5000
oov_token = '<OOV>'
pad_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(x)

word_index = tokenizer.word_index

train_sequences=tokenizer.texts_to_sequences(x)

#Padding and Truncating
train_sequences=pad_sequences(train_sequences,maxlen=100,padding=pad_type,truncating=trunc_type)

#%%
#Pre-processing of target label y
train_sequences=np.expand_dims(train_sequences,-1)
ohe=OneHotEncoder(sparse=False) 
train_subject=ohe.fit_transform(y[::,None])

#%%
#Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(train_sequences,train_subject)

#%%
#7. Model Development
embedding_size=64
model=Sequential()
model.add(Embedding(num_words,embedding_size))
model.add(Bidirectional(LSTM(embedding_size, return_sequences=True)))
model.add(LSTM(embedding_size,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dense(5,activation='softmax'))

model.summary()

#%% 
#Model Architecture
plot_model(model, show_shapes=True)

#%%
#Model Compilation
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

#%%
#Callbacks - Early Stopping and TensorBoard
LOGS_PATH=os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback= TensorBoard(log_dir=LOGS_PATH)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)

#%%
#Model Training
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,callbacks=[tensorboard_callback,early_stop_callback])
#%%
#8. Model Evaluation
y_pred=np.argmax(model.predict(X_test),axis=1)
y_true=np.argmax(y_test,axis=1)

#%%
#confusion matrix, classification report, accuracy score
print("confusion matrix: \n", confusion_matrix(y_true,y_pred))
print("classification report: \n", classification_report(y_true,y_pred))
print("accuracy score: \n", accuracy_score(y_true,y_pred))

#%%
#9. Model Saving
#Save Tokenizer
with open('saved_models.json','w') as f:
    json.dump(tokenizer.to_json(),f)


#%%
#Save one-hot encoding
with open('ohe.pkl','wb') as f:
    pickle.dump(ohe,f)

#%%
#Save deep learning model
model.save('saved_models.h5')