# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import zeros

# Load and preprocess data
df = pd.read_csv('processed_data.csv')
df = df[['reviews', 'sentiment']]
df['reviews'] = df['reviews'].str.strip()
df = df[(df['reviews'] != "") & (df['reviews'].notna())]
df['sentiment'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
df = df[df['sentiment'].isin([0, 2])]
df.reset_index(drop=True, inplace=True)

# Prepare input and output variables
X = df['reviews']
y = df['sentiment']
y = to_categorical(y)  # Convert labels to categorical

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize text
word_tokenize = Tokenizer()
word_tokenize.fit_on_texts(X_train)
X_train = word_tokenize.texts_to_sequences(X_train)
X_test = word_tokenize.texts_to_sequences(X_test)

# Determine vocabulary size and maximum sequence length
vocab_size = len(word_tokenize.word_index) + 1
maxlen = max([len(x) for x in X_train])

# Pad sequences
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Load GloVe embeddings
embeddings_dict = {}
glove_file = open('a2_glove.6B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dict[word] = vector_dimensions
glove_file.close()

# Prepare embedding matrix
embedding_matrix = zeros((vocab_size, 100))
for word, index in word_tokenize.word_index.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# Define model architecture
model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Note: The training step (model.fit) is not included in this snippet. You should add it according to your requirements.
