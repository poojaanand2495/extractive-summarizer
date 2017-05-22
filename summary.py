# coding: utf-8
import nltk
import os
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from nltk import sent_tokenize
from aeb import vectorop

import kmedoids

# 3 points in dataset
summ = open("5.txt","r")
f = summ.read()
g = sent_tokenize(f)
print(len(g))
summary = open("summary.txt","w")
op = open("med.txt","w")
text = open("sentcode.txt", "r")
data = [ ]
size = int(np.shape(vectorop)[0]/4)
for line in text:
    data.append( line.strip().split() )
data=np.asarray(data)
index = []
# distance matrix
D = pairwise_distances(data, metric='euclidean')

# split into 2 clusters
M, C = kmedoids.kMedoids(D, size)

print('medoids:')
for point_idx in M:
	print( data[point_idx], file=op )
	with open("sentcode.txt") as myFile:
		for num, line in enumerate(myFile, 1):
			if data[point_idx][1] in line:
				index.append(num)

index.sort()
print(index)

for i in index:
	print(g[i-1] ,file=summary)

print('Summary Written to File...')
print('Opening the summary now..')


	
os.system('python3 tsne1.py')			
