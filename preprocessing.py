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
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D,Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score
#pessoas não sabem acentuar palavras: remover acentos
CHARS_TO_REMOVE = [',',';',':','"',"'",'\n','\t','.','!','?',""]
# chars_to_detect = ['.','!','?',]
stemmer = nltk.stem.RSLPStemmer()

dbFile = "BigFiles/ReLi-Amado.txt"
EMBEDDING_DIM = 100
MAX_VOCAB_SIZE = 30000

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

	f_in = open(dbFile,'r')
	frase = []
	targets = []
	frase_list =[]
	value = ""
	next(f_in)
	for line in f_in:
		if not len(line)>1:
			continue 
		if line[0] is '#':
			# analyse_critica(critica)
			# critica = ""
			if len(frase)>0:
				frase_list.append(frase)
				frase = []
			continue

		parts = line.split('\t')
		word = parts[0].lower().replace('.','')


		if word in STOPWORDS or word in CHARS_TO_REMOVE:
			continue 
		# word = stemmer.stem(word)
		frase.append(word)
		value = parts[4]
	return frase_list,targets
sentences,targets = read_file(dbFile)
word2vec= loadWE()

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences) #gives each word a number
sequences = tokenizer.texts_to_sequences(sentences) #replaces each word with its index

##################################################################################
##																				##
##																				##
##	Ctrl-c Ctrl-v cnn_toxic.py para testar se está funcionando + ou menos		##
##																				##
##																				##
##################################################################################


MAX_SEQUENCE_LENGTH = 100
# MAX_VOCAB_SIZE = 20000
# EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 100

word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))

data = pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor: ',data.shape)

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
x = Conv1D(128,3,activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3,activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3,activation = 'relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128,activation = 'relu')(x)
output = Dense(1,activation = 'sigmoid')(x)

model = Model(input_,output)
model.compile(
    loss ='binary_crossentropy',
    optimizer = 'rmsprop',
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
for j in range(6):
    auc = roc_auc_score(targets[:,j],p[:,j])
    aucs.append(auc)
print(np.mean(aucs))