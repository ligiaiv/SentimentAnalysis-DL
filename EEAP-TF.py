from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os
from sklearn import metrics
from visualize_attention import attentionDisplay
from process_figshare import download_figshare, process_figshare

# tf.set_random_seed(1234)

import datetime
#pessoas não sabem acentuar palavras: remover acentos
CHARS_TO_REMOVE = [',',';',':','"',"'",'\n','\t','.','!','?',""]
BAD_STRINGS = ['http','html',':)','¬¬','=p','www','=d','p/','*-*',':d','^^','(',')','u_u','o_o','c/']
# chars_to_detect = ['.','!','?',]
possible_labels = np.array(["+","O","-"])

dbFile = "BigFiles/ReLi-Completo.txt"


def word_verify(word):
	return word in STOPWORDS or word in CHARS_TO_REMOVE or bool(re.search(r'\d', word)) or any(substring in word for substring in BAD_STRINGS)
def read_file_tweets(dbFile):
	print('Reading Dataset...')
	i=0
	# dbdf = open(dbFile,'r')
	dbdf = pd.read_csv(dbFile,sep = '|')
	targets = dbdf.ix[:,1]
	frases = dbdf.ix[:,0]
	targets = targets*(-1)+1
	targets = np.eye(3)[targets]

	frase_list = frases.apply(lambda x: [word for word in x.split() if word_verify(word)])
	
	return frase_list,np.array(targets)


def read_file_Skoob(dbFile):
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
				label = where(possible_labels == label)[0][0]
				# if value is '+':
				# 	array = [1,0,0]
				# elif value is 'O':
				# 	array = [0,1,0]
				# elif value is '-':
				# 	array = [0,0,1]
				# else:
				# 	print("problemas com targets: ",value)
				# 	array = [0,0,0]
				targets.append(label)
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
