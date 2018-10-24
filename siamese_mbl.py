import argparse
import numpy as np
from keras import backend as K
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Lambda, LSTM, Dropout, BatchNormalization, Activation
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D
from keras.preprocessing.text import one_hot

from segmentAndVectorizeDocuments_v2_new import *

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random


def splitData(X,y, max_samples_per_author=10):
    X,y=shuffle(X,y,random_state=42)
    AuthorsX={}

    for (x,y) in zip(X,y):
            y=np.where(y==1)[0][0]
            if y in AuthorsX:
                    AuthorsX[y].append(x)
            else:
                    AuthorsX[y]=[x]

#    max_samples_per_author=10

    X_left=[]
    X_right=[]
    y_lr=[]

    Done={}
    for author in AuthorsX:
        nb_texts=len(AuthorsX[author])
        nb_samples=min(nb_texts, max_samples_per_author)
        left_docs=np.array(AuthorsX[author])
        random_indexes=np.random.choice(left_docs.shape[0], nb_samples, replace=False)        
        left_sample=np.array(AuthorsX[author])[random_indexes]
        for other_author in AuthorsX:
            if  (other_author,author) in Done:
                    pass
            Done[(author,other_author)]=1
            
            right_docs=np.array(AuthorsX[other_author])
            
            nb_samples_other=min(len(AuthorsX[other_author]), max_samples_per_author)
            random_indexes_other=np.random.choice(right_docs.shape[0], nb_samples_other, replace=False)            
            right_sample=right_docs[random_indexes_other]
            
            for (l,r) in zip(left_sample,right_sample):
                    X_left.append(l)
                    X_right.append(r)            
                    if author==other_author:
                            y_lr.append(1.0)
                    else:
                            y_lr.append(0.0)
    return np.array(X_left),np.array(X_right),np.array(y_lr)

# ============================ TIMBL

def processNN(filename):
    X_left=[]
    X_right=[]
    y=[]
    FeatDict={}
    
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    
    for i in range(0,len(lines)):
        m = re.match("^([^\#].+),([A-Z]),([A-Z])\s+\{",lines[i])
        if m:
            fv=m.group(1)
            gt=m.group(2)
            pred=m.group(3)
            features_inp=fv.split(",")
            
            for f in features_inp:
                if f not in FeatDict:
                    FeatDict[f]=len(FeatDict)
            continue
        m=re.match("^\#\W(.+),\{\s+([A-Z]).+",lines[i])                    
        if m:
            fv_nn=m.group(1)
            gt_nn=m.group(2)
            features_nn=fv_nn.split(",")
            for f in features_nn:
                if f not in FeatDict:
                    FeatDict[f]=len(FeatDict)
            if fv != fv_nn:
                if pred!=gt: # error cases
                    if gt_nn==pred: # bad neighbor, majority voter: false positive NN
                        X_left.append(features_inp)
                        X_right.append(features_nn)
                        y.append(0)
                    elif gt_nn==gt:
                        X_left.append(features_inp)
                        X_right.append(features_nn)
                        y.append(1)
                else: # correct cases, bad nn outvoted: true negative NN
                    if gt == gt_nn:
                        X_left.append(features_inp)
                        X_right.append(features_nn)
                        y.append(1)
                    else:
                        X_left.append(features_inp)
                        X_right.append(features_nn)                        
                        y.append(0)
    nb_feat=len(X_left[0])
    return np.array(X_left),np.array(X_right),np.array(y), nb_feat, FeatDict 
            

            


# ===================================
    
def exponent_neg_manhattan_distance(left, right):
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


def my_one_hot(text, dict):
    return [dict[x] for x in text.split(" ")]


if __name__=="__main__":
        
    #train="/data/pan12-authorship-attribution-training-corpus-2012-03-28/"
    #test="/data/pan12-authorship-attribution-test-corpus-2012-05-24/GT/"

    train=sys.argv[1]
    test=sys.argv[2]
    
    nb_epochs=5
    nb_lstm_units=10
    
    X_train_left, X_train_right, y_train,  input_dim, FeatDict_train=processNN(train)
    X_test_left, X_test_right, y_test,  input_dim, FeatDict_test=processNN(test)

    FeatDict=mergeDictionaries(FeatDict_train, FeatDict_test)
    vocab_size=len(FeatDict)
    
    print vocab_size

    X_test_left_old=X_test_left
    X_test_right_old=X_test_right
    
    
    X_train_left_new=[]
    for x in X_train_left:
        segment=' '.join(x) #[FeatDict[v] for v in x])
        X_train_left_new.append(np.array(my_one_hot(segment,FeatDict)))
    X_train_left=np.array(X_train_left_new)

    X_train_right_new=[]
    for x in X_train_right:
        segment=' '.join(x) #[FeatDict[v] for v in x])
        X_train_right_new.append(np.array(my_one_hot(segment,FeatDict))) #pad_sequences([hashing_trick(segment, round(vocab_size*3.5))], input_dim)[0])
    X_train_right=np.array(X_train_right_new)
    

    X_test_left_new=[]
    for x in X_test_left:
        segment=' '.join(x) #[FeatDict[v] for v in x])
        X_test_left_new.append(np.array(my_one_hot(segment,FeatDict))) #pad_sequences([hashing_trick(segment, round(vocab_size*3.5))], input_dim)[0])
    X_test_left=np.array(X_test_left_new)

    X_test_right_new=[]
    for x in X_test_right:
        segment=' '.join(x)
#        print segment, my_one_hot(segment,FeatDict)
        X_test_right_new.append(np.array(my_one_hot(segment,FeatDict))) #pad_sequences([hashing_trick(segment, round(vocab_size*3.5))], input_dim)[0])
    X_test_right=np.array(X_test_right_new)
    
    print len(X_train_left), len(y_train),  len(X_test_left), len(y_test)

    
    left_input = Input(shape=(input_dim,), dtype='int32')
    right_input = Input(shape=(input_dim,), dtype='int32')

    embedding_layer = Embedding(vocab_size, 100, input_length=input_dim)


    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    shared_lstm = LSTM(nb_lstm_units)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)


    model_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    model = Model([left_input, right_input], [model_distance])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
#    exit(0)
    
    model.fit([X_train_left, X_train_right], y_train, batch_size=64, epochs=nb_epochs,
                            validation_split=0.3, verbose=2)

    #model.evaluate([X_test_left, X_test_right], y_test)
    
    loss, accuracy = model.evaluate([X_test_left, X_test_right], y_test, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

    i=0
    preds=model.predict([X_test_left, X_test_right])
    for p in preds:
        print X_test_left_old[i],X_test_right_old[i],p
        i+=1
    exit(0)
