#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import os
import subprocess
import math
#import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
sys.stderr.write("Training with IBM1 MODEL...")
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
reload(sys)  
sys.setdefaultencoding('utf8')

stemmerE = SnowballStemmer("russian")
#stemmerE.ignore_stopwords=True
stemmerG = SnowballStemmer("russian")
#stemmerG.ignore_stopwords=True

stemDict = dict()
uniqueWordsG = dict()
uniqueWordsE = dict()
for (n, (f, e)) in enumerate(bitext):
	for f_i in set(f):
		uniqueWordsG[f_i] = 0
	for e_j in set(e):
		uniqueWordsE[e_j] = 0


for xG in uniqueWordsG.keys():
 	stemDict[xG] = str(stemmerG.stem(xG))
	stemDict[str(stemmerG.stem(xG))] = str(stemmerG.stem(xG))
for xE in uniqueWordsE.keys():
	stemDict[xE] = str(stemmerE.stem(xE))
	stemDict[str(stemmerE.stem(xE))] = str(stemmerE.stem(xE))
for (n, (f, e)) in enumerate(bitext):
	for f_i in set(f):
		
		f_count[stemDict[f_i]] += 1
		for e_j in set(e):
			fe_count[(stemDict[f_i],stemDict[e_j])] += 1
	for e_j in set(e):
		e_count[stemDict[e_j]] += 1
	if n % 500 == 0:
		sys.stderr.write(".")

t = dict()

for (f_i, e_j) in fe_count.keys():
	t[f_i, e_j] = 1/float(len(e_count.keys()))   #initialize t(f|e) uniformly

num_iteration = 20
likelihood  = []
for iteration in range(num_iteration):
	
	count = defaultdict(float) # partial count of English word aligned to German
	total = defaultdict(float) #total count of English word aligned to German

  #Expectation step 
	for (n, (f, e)) in enumerate(bitext):   #for all sentence pairs (e, f) do
		for f_i in set(f):                     #for all words f_i in f do
			s_total = 0.0                        #s-total(f)=0
			for e_j in set(e):                   #for all words e_j in e do
				s_total += t[(stemDict[f_i], stemDict[e_j])]  
			for e_j in set(e):                        #for each <f,e> in count do
				count[(stemDict[f_i], stemDict[e_j])] += t[(stemDict[f_i], stemDict[e_j])]/float(s_total)                   #count[<f(i)^(n), e(j)^(n)>]++ 
				total[stemDict[e_j]] += t[(stemDict[f_i], stemDict[e_j])]/float(s_total)                         

	for (v, s) in count.keys():
		t[(stemDict[v],stemDict[s])] = (count[(stemDict[v], stemDict[s])])/float(total[stemDict[s]])  
	
for (n, (f, e)) in enumerate(bitext):
	for (i, f_i) in enumerate(f):
		max_p = 0
		best_align = 0
		for (j, e_j) in enumerate(e):
			if t[(stemDict[f_i],stemDict[e_j])] > max_p:
				max_p = t[(stemDict[f_i],stemDict[e_j])]
				best_align = j
		sys.stdout.write("%i-%i " % (i,best_align))
	sys.stdout.write("\n")
