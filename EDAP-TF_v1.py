# -*- coding: utf-8 -*-

#imports from TF-EEAP
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os
from sklearn import metrics
# from visualize_attention import attentionDisplay
# from process_figshare import download_figshare, process_figshare


#old imports from keras



from pprint import pprint
from stopwords import STOPWORDS
import nltk
import nltk.stem
import os,re
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding
from keras.layers import LSTM, Bidirectional,GlobalMaxPool1D, Dropout
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight





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

MAX_LABEL = len(possible_labels)
WORDS_FEATURE = 'words'
NUM_STEPS = 300


obs = "10 fold cross validation, categorical_crossentropy"



hparams = {'max_document_length': 60,
           'embedding_size': EMBEDDING_DIM,
           'rnn_cell_size': 128,
           'batch_size': 256,
           'attention_size': 32,
           'attention_depth': 2}

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
	
def word_verify(word):
	return word in STOPWORDS or word in CHARS_TO_REMOVE or bool(re.search(r'\d', word)) or any(substring in word for substring in BAD_STRINGS)
def read_file_tweets(dbFile):
	print('Reading Dataset...')
	i=0
	dbdf = pd.read_csv(dbFile,sep = '|')
	targets = dbdf.ix[:,1]
	frases = dbdf.ix[:,0]
	targets = targets*(-1)+1
	targets = np.eye(3)[targets]
	print(targets)

	frase_list = frases.apply(lambda x: [word for word in x.split() if word_verify(word)])
	
	return frase_list,np.array(targets)

def read_file_ReLi(dbFile):
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

		if line[0] is '#' or "[features" in line:
			continue 

		if not len(line.replace(' ',''))>1:
			# analyse_critica(critica)
			# critica = ""
			if len(frase)>0:
				# print(value)
				frase_list.append(frase)
				label = np.where(np.array(possible_labels)==value)[0][0]	
				# print(label)			
				targets.append(label)
				# print(targets)
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


sentences,targets = read_file_ReLi(dbFile)
print(targets)
print("Positivos ",(targets==0).sum(),"\nNeutros ",(targets==1).sum(),"\nNegativos ",(targets==2).sum())
print(len(sentences), " sentences were found")


word2vec= loadWE()
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences) #gives each word a number
sequences = tokenizer.texts_to_sequences(sentences) #replaces each word with its index




word2idx = tokenizer.word_index
#pprint(word2idx)
print('Found %s unique tokens.' % len(word2idx))
data = pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor: ',data.shape)

# y_ints = [y.argmax() for y in targets]
# print(y_ints)
# print(np.unique(y_ints))
# class_weights = compute_class_weight('balanced', np.unique(y_ints), y_ints)
# print(class_weights)

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

####################
#
#	TF Embed-Encode-Atent-Predict TF Aproach
#
#	based on: https://github.com/conversationai/conversationai-models/tree/master/attention-tutorial
#
######################



def embed(features):
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], 
        vocab_size=num_words, 
        embed_dim=hparams['embedding_size'])
    
    return word_vectors

def encode(word_vectors):
    # Create a Gated Recurrent Unit cell with hidden size of RNN_SIZE.
    # Since the forward and backward RNNs will have different parameters, we instantiate two seperate GRUS.
    rnn_fw_cell = tf.contrib.rnn.GRUCell(hparams['rnn_cell_size'])
    rnn_bw_cell = tf.contrib.rnn.GRUCell(hparams['rnn_cell_size'])
    
    # Create an unrolled Bi-Directional Recurrent Neural Networks to length of
    # max_document_length and passes word_list as inputs for each unit.
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, 
                                                 rnn_bw_cell, 
                                                 word_vectors, 
                                                 dtype=tf.float32, 
                                                 time_major=False)
    
    return outputs

def attend(inputs, attention_size, attention_depth):
  
  inputs = tf.concat(inputs, axis = 2)
  
  inputs_shape = inputs.shape
  sequence_length = inputs_shape[1].value
  final_layer_size = inputs_shape[2].value
  
  x = tf.reshape(inputs, [-1, final_layer_size])
  for _ in range(attention_depth-1):
    x = tf.layers.dense(x, attention_size, activation = tf.nn.relu)
  x = tf.layers.dense(x, 1, activation = None)
  logits = tf.reshape(x, [-1, sequence_length, 1])
  alphas = tf.nn.softmax(logits, dim = 1)
  
  output = tf.reduce_sum(inputs * alphas, 1)

  return output, alphas


def estimator_spec_for_softmax_classification(
    logits, labels, mode, alphas):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits),
            'attention': alphas
        })

  onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, 
                                  global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, 
                                      loss=loss, 
                                      train_op=train_op)

  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes),
      'auc': tf.metrics.auc(
          labels=labels, predictions=predicted_classes),    
  }

  logging_hook = tf.train.LoggingTensorHook({"loss" : loss}, every_n_iter=1)
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops,training_hooks = [logging_hook])


def predict(encoding, labels, mode, alphas):
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    return estimator_spec_for_softmax_classification(
          logits=logits, labels=labels, mode=mode, alphas=alphas)


def bi_rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""

  word_vectors = embed(features)
  outputs = encode(word_vectors)
  encoding, alphas = attend(outputs, 
                            hparams['attention_size'], 
                            hparams['attention_depth'])

  return predict(encoding, labels, mode, alphas)



tf.logging.set_verbosity(tf.logging.INFO)

# Rest of the function


current_time = str(int(time.time()))
model_dir = os.path.join('checkpoints', current_time)
classifier = tf.estimator.Estimator(model_fn=bi_rnn_model, 
                                    model_dir=model_dir
                                    )



# Train.
def train_model(data,targets):
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={WORDS_FEATURE: data},
	  y=targets,
	  batch_size=hparams['batch_size'],
	  num_epochs=None,
	  shuffle=True)


	classifier.train(input_fn=train_input_fn, 
	                 steps=NUM_STEPS)



# Predict.
def test_model(data,targets):
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={WORDS_FEATURE: data},
	  y=targets,
	  num_epochs=1,
	  shuffle=False)

	predictions = classifier.predict(input_fn=test_input_fn)

	y_predicted = []
	alphas_predicted = []
	for p in predictions:
	    y_predicted.append(p['class'])
	    alphas_predicted.append(p['attention'])


	scores = classifier.evaluate(input_fn=test_input_fn)
	print('Accuracy: {0:f}'.format(scores['accuracy']))
	print('AUC: {0:f}'.format(scores['auc']))

aucs_n= []
aucs_cruz = []
kf = KFold(n_splits=2)

index = 0
for train_index, test_index in kf.split(data):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = data[train_index], data[test_index]
	y_train, y_test = targets[train_index], targets[test_index]

	train_model(X_train,y_train)
	test_model(X_test,y_test)
	# confusion_matrix,acc_n,acc_cr = test_model(X_test,y_test)
	# aucs_n.append(acc_n)
	# aucs_cruz.append(acc_cr)

	