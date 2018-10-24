from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Convolution1D
from keras.layers import Flatten, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
import sys
from sklearn import preprocessing
import numpy as np
le = preprocessing.LabelEncoder()

f=open(sys.argv[1],"r")
data=[]
labels=[]
for line in f:
    words=line.rstrip().split(",")
    label=words[-1]
    data.append(' '.join(words[:-1]))
    labels.append(label)
labels=le.fit_transform(labels)


# integer encode the documents
#vocab_size = 100
#encoded_docs= [one_hot(d, vocab_size) for d in data]
#padded_docs = encoded_docs

global LexH
LexH={}

def my_one_hot(d,n):
    global Lex
    words=d.split(" ")
    resA=[]
    for w in words:
        if w not in LexH:
            LexH[w]=len(LexH.keys())
        resA.append(LexH[w])
    return resA


f=open(sys.argv[2],"r")
testdata=[]
testlabels=[]
for line in f:
    words=line.rstrip().split(",")
    label=words[-1]
    testdata.append(' '.join(words[:-1]))
    testlabels.append(label)
testlabels=le.fit_transform(testlabels)

# integer encode the documents
vocab_size = 500
encoded_docs= [my_one_hot(d, vocab_size) for d in data]
# integer encode the documents
test_docs= [my_one_hot(d, vocab_size) for d in testdata]


def load_embedding(f, vocab, embedding_dimension):
    embedding_index = {}
    f = open(f)
    n=0
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab: #only store words in current vocabulary
            coefs = np.asarray(values[1:], dtype='float32')
            if n: #skip header line
                embedding_index[word] = coefs
            n+=1
    f.close()

    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dimension))
    for word, i in vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix



embedding_dimension=100

vocab=LexH

embedding_matrix=load_embedding(sys.argv[3],vocab,embedding_dimension)

nb_epochs=int(sys.argv[4])


vocab_size=len(LexH)
max_len=len(encoded_docs[0])
embedding = Embedding(vocab_size + 1,
                            embedding_dimension,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=True)


max_length=12
model = Sequential()
model.add(embedding)
#model.add(Embedding(vocab_size, 100, input_length=max_length))
##model.add(Dense(100,activation ="relu"))
model.add(LSTM(100))
#model.add(Convolution1D(32, 30, padding="same"))
#model.add(Flatten())
model.add(Dense(len(set(labels)), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print data[0],encoded_docs[0],padded_docs[0]

model.fit(padded_docs, labels, epochs=nb_epochs, verbose=1)

weights = model.layers[0].get_weights()[0]

#print weights[10]
#loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

padded_testdocs = pad_sequences(test_docs, maxlen=max_length, padding='post')
loss, accuracy = model.evaluate(padded_testdocs, testlabels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
