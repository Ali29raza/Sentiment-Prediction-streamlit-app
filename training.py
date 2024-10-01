# initial imports
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv('Twitter_Data.csv')  # Ensure you provide the correct path to your CSV file

# Preprocess text data
max_features = 5000 # accounts for the toal volabulary to be considered (frequency wise)
max_length = 100 # accounts for the max sequence length per document (max words per document) (it will be bigger if the document is bigger like a paragraph or book)

# There are some missing values in the 'text' column so lets drop them
data.dropna(subset = ['text'], inplace = True)  # only drop where text is missing

# tokenize the text column
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['text'])
X = tokenizer.texts_to_sequences(data['text']) 
X = pad_sequences(X, maxlen=max_length) # if the max_length of document is less than 100 it will pad (add) zeros 

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['sentiment'])  # label encoder does the oridnal encoding

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim = max_features,output_dim =  128))
# model.add(SpatialDropout1D(0.2))   # 20% of the data this is for avoiding over fitting it will randomly drop some neurons with every itteration.
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile model
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model on the entire dataset
model.fit(X, y, epochs=5, batch_size=64)

# Save the model and the tokenizer
#model.save('sentiment_model.h5')
model.save('sentiment_model.keras', include_optimizer=False)

joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and Tokenizer saved.")
