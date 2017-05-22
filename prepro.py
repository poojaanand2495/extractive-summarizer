import nltk
import re
import os
import sys
import numpy
import numpy as np
import matplotlib.pylab as plt
numpy.set_printoptions(threshold=numpy.nan)
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

f = open('5.txt', 'r')
op_file = open('output1.txt', 'w')
g = f.read()
y = sent_tokenize(g)
print(y, file=op_file)

w = [word_tokenize(t) for t in y]
port = []

print(w, file=op_file)

porter = PorterStemmer()
for i in range(0,len(y)):
	port.append([porter.stem(q) for q in w[i]])

print(port, file=op_file)
a=[]
for i in range(0, len(port)):
	for t in port[i]:
		a.append(''.join(t))

print(a, file=op_file)


b = ' '.join(a)

print(b, file=op_file)

r = sent_tokenize(b)

vec = CountVectorizer(r, stop_words=u'english')
vectorop = vec.fit_transform(r)


op2_file = open('op3.txt','w')
l = vectorop.toarray()
#m = l.tolist()
np.savetxt("op3.txt", l, newline='\n')

#print("".join(re.sub('[\[\]]', '', np.array_str(l))), file=op2_file)
#print(, sep='\n', file=op2_file)

#os.system('python3 tsne1.py')
