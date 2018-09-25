# -*- coding: utf-8 -*-
from pprint import pprint
from stopwords import STOPWORDS
import nltk
import nltk.stem
import os,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding
from keras.layers import LSTM, Bidirectional,GlobalMaxPool1D, Dropout
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import roc_auc_score
#pessoas não sabem acentuar palavras: remover acentos
CHARS_TO_REMOVE = [',',';',':','"',"'",'\n','\t','.','!','?',""]
# chars_to_detect = ['.','!','?',]
stemmer = nltk.stem.RSLPStemmer()

dbFile = "BigFiles/ReLi-Completo.txt"
EMBEDDING_DIM = 300
MAX_VOCAB_SIZE = 30000
possible_labels = ["+","O","-"]
def loadWE():
	print('Loading word vectors...')
	word2vec = {}
	here= os.path.dirname(os.path.realpath(__file__))
	print(here)

	with open(here + ('/BigFiles/glove_s%s.txt' % EMBEDDING_DIM)) as f:
		for line in f:
			values = line.split()
			len_word = (len(values)-(EMBEDDING_DIM))

			word = ' '.join(values[0:(len(values)-(EMBEDDING_DIM))])
			vec = np.asarray(values[len_word:],dtype = 'float32')
			word2vec[word] = vec

	print('Found %s word vectors.' % len(word2vec))
	return word2vec
	

def read_file(dbFile):
	print('Reading Dataset...')
	i=0
	f_in = open(dbFile,'r')
	frase = []
	targets = []
	frase_list =[]
	value = ""
	next(f_in)
	for line in f_in:
		#i+=1
		#print(i," ",line)

		if not len(line.replace(' ',''))>1:
			continue 
		if line[0] is '#' or line[0] is '[':
			# analyse_critica(critica)
			# critica = ""
			if len(frase)>0:
				# print(value)
				frase_list.append(frase)
				if value is '+':
					array = [1,0,0]
				elif value is 'O':
					array = [0,1,0]
				elif value is '-':
					array = [0,0,1]
				else:
					print("problemas com targets: ",value)
					array = [0,0,0]
				targets.append(array)
				# print(targets)
				# print(array)
				
				frase = []
			continue

		parts = line.split('\t')
		word = parts[0].lower().replace('.','')


		if word in STOPWORDS or word in CHARS_TO_REMOVE:
			continue 
		# word = stemmer.stem(word)
		frase.append(word)
		value = parts[4]
	return frase_list,np.array(targets)
sentences,targets = read_file(dbFile)
print(len(sentences), " sentences were found")
word2vec= loadWE()
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences) #gives each word a number
sequences = tokenizer.texts_to_sequences(sentences) #replaces each word with its index

##################################################################################
##																				##
##																				##
##	Ctrl-c Ctrl-v cnn_toxic.py para testar se está funcionando + ou menos		##
##	Modificado para usar próprias redes											##
##																				##
##																				##
##################################################################################


MAX_SEQUENCE_LENGTH = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

word2idx = tokenizer.word_index
pprint(word2idx)
print('Found %s unique tokens.' % len(word2idx))
data = pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor: ',data.shape)
# quit()


#prepare embedding matrix

print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word,i in word2idx.items():
	if i<MAX_VOCAB_SIZE:
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector


# load pre-trained word embedding into an Embedding layer
# trainable = False so the embeddings are fixed

embedding_layer=Embedding(
    num_words,
    EMBEDDING_DIM,
    weights = [embedding_matrix],
    input_length = MAX_SEQUENCE_LENGTH,
    trainable = False
)

print('Building model ...')

# train a 1D convnet with global maxpooling
input_ = Input(shape = (MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Bidirectional(LSTM(30,return_sequences = True))(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels),activation = 'sigmoid')(x)

model = Model(input_,output)
model.compile(
    loss ='binary_crossentropy',
    optimizer = Adam(lr = 0.01),
    metrics = ['accuracy']
)

print('training model...')
r = model.fit(
    data,
    targets,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split = VALIDATION_SPLIT
    )
plt.plot(r.history['loss'],label = 'loss')
plt.plot(r.history['val_acc'],label = 'val_acc')
plt.legend()
plt.show()

p = model.predict(data)
aucs = []
for j in range(3):
    auc = roc_auc_score(targets[:,j],p[:,j])
    aucs.append(auc)
print(np.mean(aucs))



