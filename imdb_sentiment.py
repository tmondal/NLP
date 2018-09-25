# Import preprocessed database
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

vocabulary = 5000

(x_train, y_train), (X_test, Y_test) = imdb.load_data(num_words = vocabulary)
print('Training samples : {}, {} Test samples'.format(len(x_train), len(X_test)))

max_words = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

embedding_size=32
my_model=Sequential()
my_model.add(Embedding(vocabulary, embedding_size, input_length=max_words))
my_model.add(LSTM(100))
my_model.add(Dense(1, activation='sigmoid'))

print(my_model.summary())


my_model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

batch_size = 64
num_epochs = 100

X_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]

my_model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)


scores = my_model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', scores[1])