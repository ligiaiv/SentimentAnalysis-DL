from pprint import pprint
from stopwords import STOPWORDS
import nltk
import nltk.stem
#pessoas não sabem acentuar palavras: remover acentos
CHARS_TO_REMOVE = [',',';',':','"',"'",'\n','\t','.','!','?',""]
# chars_to_detect = ['.','!','?',]
stemmer = nltk.stem.RSLPStemmer()

file_in = "ReLi-Amado.txt"

f_in = open(file_in,'r')
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
	word = stemmer.stem(word)
	frase.append(word)
	value = parts[4]

	# critica.append(line) 
pprint(frase_list)