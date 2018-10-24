from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt

import numpy as np
import sys

import random
import re
import codecs


def save_embeddings(save_filepath, weights, vocabulary):
        rev = {v:k for k, v in vocabulary.iteritems()}
	with codecs.open(save_filepath, "w") as f:
		f.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
                for index in sorted(rev.iterkeys()):
                        word=rev[index]
#                        word=[str(x) for x in word]
 #                       word='_'. join(word)
			f.write(word + " ")
                      	for i in xrange(len(weights[index])):
		            f.write(str(weights[index][i]) + " ")
	                f.write("\n")  


def getLines(f):
    lines = [line.rstrip() for line in open(f)]
    return lines



def generator(target,context, labels, batch_size):
    batch_target = np.zeros((batch_size, 1))
    batch_context = np.zeros((batch_size, 1))
    batch_labels = np.zeros((batch_size,1))

    while True:
        for i in range(batch_size):
            index= random.randint(0,len(target)-1)
            batch_target[i] = target[index]
            batch_context[i]=context[index]
            batch_labels[i] = labels[index]
        yield [batch_target,batch_context], [batch_labels]        


# ========================= MVDM ==========================================

ValueIdentifierDict={}

def getValueIdentifier(v):
    global ValueIdentifierDict
    if v not in ValueIdentifierDict:
        ValueIdentifierDict[v]=len(ValueIdentifierDict)
    return ValueIdentifierDict[v]


def indexFeatureVectors(filename):
    ValueIndex={} # i-> values
    ValueClassIndex={} # (v,i) => labels
    LabelIndex={}
    RevIndex={}
    lines=getLines(filename)
    for line in lines:
        values=line.split(",")
        label=values[-1:][0]
        LabelIndex[label]=1
        values=values[:-1]
        i=0
        for v in values:
            oldV=v    
            v=getValueIdentifier(v)
            RevIndex[v]=oldV
            if i not in ValueIndex:
                ValueIndex[i]=[v]
            elif v not in ValueIndex[i]:
                ValueIndex[i].append(v)            
            if v not in ValueClassIndex:
                ValueClassIndex[v]=[label]
            elif label not in ValueClassIndex[v]:
                ValueClassIndex[v].append(label)
            i+=1
    return len(ValueIdentifierDict),ValueIndex,ValueClassIndex, LabelIndex, RevIndex


def feature_class_couples(ValueIndex, ValueClassIndex, LabelIndex, RevIndex):
    couples=[]
    labels=[]
    for i in range(len(ValueIndex)):
        for v1 in ValueIndex[i]:
            for v2 in ValueIndex[i]:
                    if v1==v2:
                            continue
                    intersection_len=len(list(set(ValueClassIndex[v1]) & set(ValueClassIndex[v2])))
                    if intersection_len==0:
                            couples.append([v1,v2])
                            labels.append(0)
                    else:
                        for i in range(intersection_len):
                            couples.append([v1,v2])
                            labels.append([1])

    return couples,labels


def load_embedding(f, embedding_dimension):
    embedding_index = {}
    f = open(f)
    n=0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if n: #skip header line
                embedding_index[word] = coefs
        n+=1
    f.close()

    return embedding_index


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# ========================= MVDM ==========================================                
                
                   

                
def collect_data(textFile):
    couples=[]
    labels=[]
    
    (vocab_size,ValueIndex, ValueClassIndex,LabelIndex, RevIndex)=indexFeatureVectors(textFile) # every line a labeled feature vector
    
    couples,labels=feature_class_couples(ValueIndex, ValueClassIndex, LabelIndex, RevIndex)

    return vocab_size, couples,labels            



window_size = 3
vector_dim = 100
epochs = 1000

vocab_size,couples,labels=collect_data(sys.argv[1])


word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

dot_product = merge([target, context], mode='dot', dot_axes=1)
dot_product = Reshape((1,))(dot_product)
output = Dense(1, activation='softmax')(dot_product) #sigmoid

#output = Dense(len(set(labels)), activation='sigmoid')(dot_product)

model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])

#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])

print model.summary()

epochs=int(sys.argv[2])

model.fit_generator(generator(word_target, word_context,labels,100), steps_per_epoch=100,epochs=epochs)

save_embeddings("embedding.txt", embedding.get_weights()[0], ValueIdentifierDict)

model=load_embedding("embedding.txt", 100)
#tsne_plot(model)

exit(0)

