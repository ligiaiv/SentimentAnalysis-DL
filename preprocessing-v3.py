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
import datetime
#pessoas não sabem acentuar palavras: remover acentos
CHARS_TO_REMOVE = [',',';',':','"',"'",'\n','\t','.','!','?',""]
BAD_STRINGS = ['http','html',':)','¬¬','=p','www','=d','p/','*-*',':d','^^','(',')','u_u','o_o','c/']
# chars_to_detect = ['.','!','?',]
stemmer = nltk.stem.RSLPStemmer()

dbFile = "BigFiles/ReLi-Completo.txt"
EMBEDDING_DIM = 100
MAX_VOCAB_SIZE = 30000
M = 10 #nº de camadas
possible_labels = ["+","O","-"]
MAX_SEQUENCE_LENGTH = 100
VALIDATION_SPLIT = 0.2
TRAIN_TEST_SPLIT = 0.7
BATCH_SIZE = 128
EPOCHS = 20

def rand_shuffle(data,targets):
	# print(target.shape)
	joint = np.concatenate([data,targets],axis = 1)
	np.random.shuffle(joint)
	# print(joint)
	n_data = joint[:,:-3]
	n_targets = joint[:,-3:]
	# print("n_data: \n",n_data)
	# print("n_target: \n",n_target)
	# # oi = np.array([data,target],axis = 1)

	# print(type(n_data))
	return n_data,n_targets

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


		if word in STOPWORDS or word in CHARS_TO_REMOVE or bool(re.search(r'\d', word)) or any(substring in word for substring in BAD_STRINGS):#remove stopwords, ponctuation, numbers and bad strings
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



word2idx = tokenizer.word_index
#pprint(word2idx)
print('Found %s unique tokens.' % len(word2idx))
data = pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor: ',data.shape)


#prepare embedding matrix

print('Filling pre-trained embeddings...')
words_not_found = []

num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word,i in word2idx.items():
	if i<MAX_VOCAB_SIZE:
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			words_not_found.append(word)
print("total de palavras: ",len(word2idx),"\tpalavras não encontradasno embedding: ",len(words_not_found))
with open("BigFiles/words_not_in_emb.txt",'w') as fout:
	fout.write('\n'.join(words_not_found))


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
x = Bidirectional(LSTM(M,return_sequences = True))(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels),activation = 'sigmoid')(x)

model = Model(input_,output)
model.compile(
    loss ='binary_crossentropy',
    optimizer = Adam(lr = 0.01),
    metrics = ['accuracy']
)

def train_model(data,targets,train_split):



	print('training model...')
	r = model.fit(
	    data,
	    targets,
	    batch_size = BATCH_SIZE,
	    epochs = EPOCHS,
	    validation_split = VALIDATION_SPLIT
	    )
	


def test_model(data,targets,train_split):
	print("testing model...")

	p = model.predict(data)
	print(len(p)," ",len(targets))
	p_bool=p.max(axis=1,keepdims=1) == p
	result = (targets == p_bool).all(axis=1)
	print("result:  ",result)
	auc_n = result.sum()/test_split
	print('auc_n= ',auc_n )

	# plt.plot(r.history['loss'],label = 'loss')
	# plt.plot(r.history['val_acc'],label = 'val_acc')
	# plt.legend()
	# plt.show()

	target_test = targets[train_split:]
	posneg_position = target_test[:,1] == 0
	total_posneg = len(target_test)-target_test[:,1].sum()
	print(total_posneg)
	# total_posneg = posneg_position.astype(np.int32).sum()

	auc_cruz = (result*posneg_position).sum()/total_posneg
	print('auc_cruz = ',auc_cruz )

	return auc_n,auc_cruz


aucs = []
data_size = data.shape[0]


data,targets =rand_shuffle(data,targets)
train_split = int(np.ceil(data_size*TRAIN_TEST_SPLIT))
test_split = data_size-train_split
print("train :",train_split)
data_train = data[:train_split]
targets_train =  targets[:train_split]
data_test = data[train_split:]
targets_test = targets[train_split:]
for i in range(5):
	data_tr,targets_tr =rand_shuffle(data_train,targets_train)


	train_model(data_tr,targets_tr,train_split)
	

result = test_model(data_test,targets_test,train_split)
aucs.append(result)

aucs_med = list(map(sum, zip(result)))

now = datetime.datetime.now()


with open("report",'a') as outfile:
	outfile.write("\n*************"+now.strftime("%Y-%m-%d %H:%M")+"*************")
	outfile.write("\nEMBEDDING_DIM"+str(EMBEDDING_DIM))
	outfile.write("\nMAX_VOCAB_SIZE "+str(MAX_VOCAB_SIZE))
	outfile.write("\nM "+str(M))
	outfile.write("\npossible_labels "+','.join(possible_labels))
	outfile.write("\nMAX_SEQUENCE_LENGTH "+str(MAX_SEQUENCE_LENGTH))
	outfile.write("\nVALIDATION_SPLIT "+str(VALIDATION_SPLIT))
	outfile.write("\nTRAIN_TEST_SPLIT "+str(TRAIN_TEST_SPLIT))
	outfile.write("\nBATCH_SIZE "+str(BATCH_SIZE))
	outfile.write("\nEPOCHS  "+str(EPOCHS))

	outfile.write("\nauc_n media  "+str(aucs_med[0]))
	outfile.write("\nauc_cruz media  "+str(aucs_med[1]))
# plt.plot(r.history['loss'],label = 'loss')
# plt.plot(r.history['val_acc'],label = 'val_acc')
# plt.legend()
# plt.show()

# aucs = []
# for j in range(3):
#     auc = roc_auc_score(targets[:,j],p[:,j])
#     aucs.append(auc)
# print(np.mean(aucs))


